# langchain-LLM
在 Windows 下：需要安装 Microsoft C++ Build Tools

在 CPU 下：需要安装 gcc 与 openmp，安装 [TDM-GCC](https://jmeubank.github.io/tdm-gcc/) 时勾选 `openmp`

No compiled kernel found 报错，运行如下命令：
```shell
gcc -fPIC -pthread -fopenmp -std=c99 quantization_kernels.c -shared -o quantization_kernels.so
gcc -fPIC -pthread -fopenmp -std=c99 quantization_kernels_parallel.c -shared -o quantization_kernels_parallel.so
```