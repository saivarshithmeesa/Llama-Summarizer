# LLama-2-PDF-summarizer-QA
### Unclocking the power of LLama-2 for local Multi-Document Summarization and Chat with multiple Documents
## SETUP
### Download CudaToolKit-[https://developer.nvidia.com/cuda-11-8-0-download-archive]
### Download CMake-[https://cmake.org/download/]
### NOTE:-Check if cudatoolkit and cmake paths are added to enviroment variables
#### After that pip install this
```
pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117 
```
```
pip install langchain streamlit PyPDF2 faiss-cpu transformers huggingface_hub
```
### Download LLama-2 8-bit Quantized model or lesser based on your PC Computational power from :-[https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main]
## Note:-Donot forget to create a data folder in the directory where uploaded pdf will be saved
## Run
```
python -m streamlit run index.py
```
## For more Information please check the documentation : [https://drive.google.com/file/d/1PVaZ-Xd8h5m1gliCpuloLRSPGugFKgjy/view?usp=sharing]
