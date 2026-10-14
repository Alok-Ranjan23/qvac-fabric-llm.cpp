"""Diagnostic branch only: preserve SET_ROWS math/tolerance, capture and replay raw inputs."""
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / 'tests/test-backend-ops.cpp'
s = path.read_text()
marker = '// GGML_OP_ROPE + GGML_OP_VIEW + GGML_OP_SET_ROWS'
assert s.count(marker) == 1
harness = r'''
// Diagnostic-only corpus replay for the exact failing CI case.
struct diagnostic_set_rows : test_set_rows {
    const int corpus_id;
    diagnostic_set_rows(int id)
        : test_set_rows(GGML_TYPE_F16, GGML_TYPE_PQ4_0_64, GGML_TYPE_I64,
                        {192, 3, 1, 1}, {2, 3}, 2, false), corpus_id(id) {}

    std::string vars() override {
        return test_set_rows::vars() + ",corpus=" + std::to_string(corpus_id);
    }

    void initialize_tensors(ggml_context * ctx) override {
        const char * dir = std::getenv("CM2_CORPUS_DIR");
        const char * mode = std::getenv("CM2_CORPUS_MODE");
        GGML_ASSERT(dir != nullptr && mode != nullptr);
        const bool capture = std::strcmp(mode, "capture") == 0;
        GGML_ASSERT(capture || std::strcmp(mode, "replay") == 0);
        const std::string path = std::string(dir) + "/" + std::to_string(corpus_id) + ".bin";
        std::fstream file(path, std::ios::binary | (capture ? std::ios::out : std::ios::in));
        GGML_ASSERT(file.is_open());
        if (capture) {
            // Use the unmodified test's initialization, including random row indices.
            test_set_rows::initialize_tensors(ctx);
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op)) {
                continue;
            }
            const uint64_t size = ggml_nbytes(t);
            const int32_t type = t->type;
            std::vector<uint8_t> bytes(size);
            if (capture) {
                ggml_backend_tensor_get(t, bytes.data(), 0, size);
                file.write(reinterpret_cast<const char *>(&type), sizeof(type));
                file.write(reinterpret_cast<const char *>(&size), sizeof(size));
                file.write(reinterpret_cast<const char *>(bytes.data()), size);
            } else {
                int32_t saved_type = -1;
                uint64_t saved_size = 0;
                file.read(reinterpret_cast<char *>(&saved_type), sizeof(saved_type));
                file.read(reinterpret_cast<char *>(&saved_size), sizeof(saved_size));
                GGML_ASSERT(saved_type == type && saved_size == size);
                file.read(reinterpret_cast<char *>(bytes.data()), size);
                GGML_ASSERT(file.good());
                ggml_backend_tensor_set(t, bytes.data(), 0, size);
                std::vector<uint8_t> actual(size);
                ggml_backend_tensor_get(t, actual.data(), 0, size);
                GGML_ASSERT(actual == bytes); // Require byte-identical device inputs.
            }
            GGML_ASSERT(file.good());
        }
        if (!capture) {
            GGML_ASSERT(file.peek() == std::char_traits<char>::eof());
        }
    }
};

'''
s = s.replace(marker, harness + marker)
marker2 = '    filter_test_cases(test_cases, params_filter);'
assert s.count(marker2) == 1
s = s.replace(marker2, r'''
    if (const char * count_str = std::getenv("CM2_CORPUS_COUNT")) {
        const int count = std::atoi(count_str);
        GGML_ASSERT(mode == MODE_TEST && count > 0 && count <= 4096);
        test_cases.clear();
        for (int i = 0; i < count; ++i) {
            test_cases.emplace_back(new diagnostic_set_rows(i));
        }
    }
''' + marker2)
path.write_text(s)
