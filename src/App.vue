<script setup>
import { ref } from "vue";
import { env } from "@huggingface/transformers";
import { ImgComparisonSlider } from "@img-comparison-slider/vue";

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;

const isLoading = ref(false);
const isModelLoading = ref(true);
const status = ref("初始化中...");
const images = ref([]);
const selectedImageIndex = ref(null);
const error = ref(null);
const isDragging = ref(false);
const sliderPosition = ref(50);
const isSliderDragging = ref(false);

let imageProcessor = null;

function initializeWorker() {
  imageProcessor = new Worker(new URL("./workers/imageProcessor.js", import.meta.url), { type: "module" });

  imageProcessor.onmessage = e => {
    const { type, result, error: workerError } = e.data;

    switch (type) {
      case "modelLoaded":
        isModelLoading.value = false;
        status.value = "准备就绪";
        break;
      case "processComplete":
        const { processedImage, index } = result;
        if (images.value[index]) {
          images.value[index].processed = processedImage;
          images.value[index].status = "completed";
        }
        break;
      case "error":
        error.value = workerError;
        status.value = "处理失败";
        break;
    }
  };

  imageProcessor.onerror = err => {
    error.value = `Worker error: ${err.message}`;
    status.value = "处理失败";
  };

  // Initialize the model in worker
  imageProcessor.postMessage({ type: "init" });
}

async function handleFiles(files) {
  if (!files.length) return;

  try {
    isLoading.value = true;
    error.value = null;

    // Process each file
    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      // Check if file is an image
      if (!file.type.startsWith("image/")) {
        error.value = `${file.name} 不是图片文件`;
        continue;
      }

      status.value = `处理第 ${i + 1}/${files.length} 张图片...`;

      // Create image entry
      const imageEntry = {
        original: null,
        processed: null,
        originalLoaded: false,
        name: file.name,
        size: formatFileSize(file.size),
        status: "processing",
      };
      const currentIndex = images.value.length;
      images.value.push(imageEntry);

      // Set selected index to first image if none selected
      if (selectedImageIndex.value === null) {
        selectedImageIndex.value = currentIndex;
      }

      // Display original image
      const reader = new FileReader();
      await new Promise(resolve => {
        reader.onload = e => {
          imageEntry.original = e.target.result;
          resolve();
        };
        reader.readAsDataURL(file);
      });

      // Process image in worker
      imageProcessor.postMessage({
        type: "process",
        data: {
          image: file,
          index: currentIndex,
        },
      });
    }
  } catch (e) {
    error.value = e.message;
    status.value = "处理失败";
  } finally {
    isLoading.value = false;
    isDragging.value = false;
  }
}

function handleFileUpload(event) {
  handleFiles(event.target.files);
}

function handleDragEnter(e) {
  e.preventDefault();
  e.stopPropagation();
  isDragging.value = true;
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  if (!e.currentTarget.contains(e.relatedTarget)) {
    isDragging.value = false;
  }
}

function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
}

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  isDragging.value = false;

  const files = [...e.dataTransfer.files];
  handleFiles(files);
}

function handleSliderMove(event) {
  if (!isSliderDragging.value) return;

  const container = event.currentTarget;
  const rect = container.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

  sliderPosition.value = percentage;
}

function handleSliderStart() {
  isSliderDragging.value = true;
}

function handleSliderEnd() {
  isSliderDragging.value = false;
}

// Delete a single image
function deleteImage(index, event) {
  event.stopPropagation();
  images.value = images.value.filter((_, i) => i !== index);
  if (selectedImageIndex.value === index) {
    selectedImageIndex.value = images.value.length > 0 ? 0 : null;
  } else if (selectedImageIndex.value > index) {
    selectedImageIndex.value--;
  }
}

// Clear all images
function clearAllImages() {
  images.value = [];
  selectedImageIndex.value = null;
}

// Export all processed images
async function exportAllProcessed() {
  const processedImages = images.value.filter(img => img.status === "completed" && img.processed);

  if (processedImages.length === 0) {
    error.value = "没有可导出的已处理图片";
    return;
  }

  for (const img of processedImages) {
    const link = document.createElement("a");
    link.href = img.processed;
    link.download = `processed_${img.name}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    await new Promise(resolve => setTimeout(resolve, 100)); // Small delay between downloads
  }
}

// Utility function to format file size
function formatFileSize(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// Initialize worker when component mounts
initializeWorker();
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <h1 class="text-4xl font-bold mb-2 text-center text-gray-900">AI 移除背景</h1>
      <p class="text-center text-gray-600 mb-8">快速移除图片背景，支持批量处理</p>

      <div v-if="error" class="mb-6 mx-auto max-w-md">
        <div class="bg-red-50 border-l-4 border-red-400 p-4">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm text-red-700">{{ error }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Loading Status -->
      <div v-if="isLoading || isModelLoading" class="fixed top-4 right-4 z-50">
        <div class="inline-flex items-center px-4 py-2 bg-white rounded-lg shadow-sm">
          <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
          <p class="text-gray-700">{{ status }}</p>
        </div>
      </div>

      <div class="flex flex-col lg:flex-row gap-8">
        <!-- Left Column: Upload and Image List -->
        <div class="w-full lg:w-1/3 space-y-6">
          <!-- Upload Section -->
          <div class="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow relative cursor-pointer group" @dragenter="handleDragEnter" @dragleave="handleDragLeave" @dragover="handleDragOver" @drop="handleDrop" @click="$refs.fileInput.click()">
            <input type="file" accept="image/*" multiple class="hidden" @change="handleFileUpload" ref="fileInput" />

            <!-- Drag overlay -->
            <div v-if="isDragging" class="absolute inset-0 bg-blue-500 bg-opacity-10 border-2 border-blue-500 border-dashed rounded-xl z-10 flex items-center justify-center">
              <div class="text-blue-500 font-medium">
                <svg class="mx-auto h-12 w-12 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                释放鼠标上传图片
              </div>
            </div>

            <div class="p-6" :class="{ 'pointer-events-none': isDragging }">
              <div class="space-y-4">
                <div class="text-center">
                  <svg class="mx-auto h-12 w-12 text-gray-400 group-hover:text-blue-500 transition-colors" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <div class="mt-4">
                    <span class="inline-flex items-center px-4 py-2 shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 group-hover:bg-blue-700 transition-colors" :class="{ 'opacity-50 cursor-not-allowed': isModelLoading }">
                      <svg class="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0l-4 4m4-4v12" />
                      </svg>
                      选择图片
                    </span>
                  </div>
                  <p class="mt-2 text-sm text-gray-500">或将图片拖拽到此处</p>
                  <p class="mt-1 text-xs text-gray-400">支持 PNG、JPG、JPEG 格式</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Image List -->
          <div class="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow">
            <div class="p-4 border-b border-gray-200">
              <div class="flex justify-between items-center">
                <h3 class="text-lg font-medium text-gray-900">图片列表</h3>
                <div class="space-x-2">
                  <button @click="exportAllProcessed" class="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 transition-colors" :disabled="!images.length">导出全部</button>
                  <button @click="clearAllImages" class="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors" :disabled="!images.length">清空</button>
                </div>
              </div>
            </div>
            <div class="divide-y divide-gray-200 max-h-[500px] overflow-y-auto">
              <div 
                v-for="(image, index) in images" 
                :key="index" 
                @click="!isModelLoading && (selectedImageIndex = index)" 
                class="p-4 hover:bg-gray-50" 
                :class="{ 
                  'cursor-pointer': !isModelLoading,
                  'cursor-not-allowed': isModelLoading,
                  'bg-blue-50': selectedImageIndex === index 
                }"
              >
                <div class="flex items-center justify-between">
                  <div class="min-w-0 flex-1">
                    <div class="flex justify-between items-center">
                      <p class="text-sm font-medium text-gray-900 truncate">{{ image.name }}</p>
                      <button @click="deleteImage(index, $event)" class="p-1 text-gray-400 hover:text-red-500 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                        </svg>
                      </button>
                    </div>
                    <div class="mt-1 flex items-center text-xs text-gray-500 space-x-2">
                      <span>{{ image.size }}</span>
                      <span>•</span>
                      <span v-if="image.status === 'processing'" class="inline-flex items-center">
                        <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        处理中...
                      </span>
                      <span v-else class="inline-flex items-center text-green-600">
                        <svg class="-ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                        </svg>
                        处理完成
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: Image Preview -->
        <div class="flex-1 bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow min-h-[600px] flex flex-col">
          <div v-if="selectedImageIndex !== null && images[selectedImageIndex]" class="h-full flex flex-col">
            <ImgComparisonSlider :key="selectedImageIndex" v-if="selectedImageIndex !== null && images[selectedImageIndex].processed">
              <img slot="first" style="width: 100%" :src="images[selectedImageIndex].original" />
              <img slot="second" style="width: 100%" :src="images[selectedImageIndex].processed" />
            </ImgComparisonSlider>
          </div>

          <!-- No Image Selected State -->
          <div v-else class="h-full flex items-center justify-center">
            <div class="text-center text-gray-500">
              <svg class="mx-auto h-12 w-12 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p class="mt-2">选择图片以预览</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
