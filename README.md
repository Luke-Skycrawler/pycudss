#### cudss CUDA bindings 

bind nvidia [cudss](https://developer.nvidia.com/cudss) solver to python. 

### Setup 

prerequisite: cuda, cudss

copy the following dlls into this folder:
```
cublas64_12.dll
cublasLt64_12.dll
cudart64_12.dll
cudss64_0.dll
cusparse64_12.dll
libiomp5md.dll
nvJitLink_120_0.dll
python310.dll
```

```
pip install -e .
```
run `test.py` to verify successful instalation: 
```python
python test.py 
```


### Known Issues 

- Can only handle symmetric positive semi definite matrix.
- Only supports double


### CMake Debugging Command

```
set CUDSS_INSTALL_DIR="C:\Program Files\NVIDIA cuDSS\v0.5"
cmake -B build -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake -Dcudss_DIR=%CUDSS_INSTALL_DIR%/lib/12/cmake/cudss -DPYTHON_EXECUTABLE=d:\miniconda3\python.exe -DCUDSS_INSTALL_DIR=%CUDSS_INSTALL_DIR%
```


