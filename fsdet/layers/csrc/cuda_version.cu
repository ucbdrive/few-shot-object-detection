#include <cuda_runtime_api.h>

namespace fsdet {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace fsdet
