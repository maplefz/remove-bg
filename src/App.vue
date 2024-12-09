<script setup>
import { ref } from "vue";
import { AutoModel, AutoProcessor, env, RawImage } from "@huggingface/transformers";

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;

const isLoading = ref(false);
const isModelLoading = ref(true);
const status = ref("初始化中...");
const images = ref([]); // Store multiple images
const selectedImageIndex = ref(null); // Track selected image
const error = ref(null);

let model = null;
let processor = null;

async function initializeModel() {
  try {
    status.value = "正在加载模型...";
    model = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
      config: { model_type: "custom" },
      device: "webgpu",
    });
    status.value = "正在加载处理器...";
    processor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
      config: {
        do_normalize: true,
        do_pad: false,
        do_rescale: true,
        do_resize: true,
        image_mean: [0.5, 0.5, 0.5],
        feature_extractor_type: "ImageFeatureExtractor",
        image_std: [1, 1, 1],
        resample: 2,
        rescale_factor: 0.00392156862745098,
        size: { width: 1024, height: 1024 },
      },
    });
    status.value = "准备就绪";
    isModelLoading.value = false;
  } catch (e) {
    error.value = "模型加载失败: " + e.message;
    status.value = "初始化失败";
    isModelLoading.value = false;
  }
}

async function handleFileUpload(event) {
  const files = event.target.files;
  if (!files.length) return;

  try {
    isLoading.value = true;
    error.value = null;

    // Initialize model if needed
    if (!model || !processor) {
      status.value = "初始化模型...";
      await initializeModel();
    }

    // Process each file
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      status.value = `处理第 ${i + 1}/${files.length} 张图片...`;

      // Create image entry
      const imageEntry = {
        original: null,
        processed: null,
        name: file.name,
      };
      images.value.push(imageEntry);

      // Display original image
      const reader = new FileReader();
      await new Promise(resolve => {
        reader.onload = e => {
          imageEntry.original = e.target.result;
          resolve();
        };
        reader.readAsDataURL(file);
      });

      // Process image
      const image = await RawImage.fromBlob(file);
      const { pixel_values } = await processor(image);
      const { output } = await model({ input: pixel_values });

      // Generate final image
      const mask = await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(image.width, image.height);
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image.toCanvas(), 0, 0);
      const pixelData = ctx.getImageData(0, 0, image.width, image.height);

      for (let i = 0; i < mask.data.length; ++i) {
        pixelData.data[4 * i + 3] = mask.data[i];
      }

      ctx.putImageData(pixelData, 0, 0);
      imageEntry.processed = canvas.toDataURL();
    }

    status.value = "处理完成";
  } catch (e) {
    error.value = "处理图片失败: " + e.message;
    status.value = "处理失败";
  } finally {
    isLoading.value = false;
  }
}

// Initialize model when component mounts
initializeModel();
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
          <div class="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <label class="block text-gray-700 font-medium mb-3" for="image">选择图片</label>
            <div class="relative">
              <input type="file" multiple accept="image/*" @change="handleFileUpload" :disabled="isLoading || isModelLoading" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed" />
            </div>
          </div>

          <!-- Image List -->
          <div class="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <h3 class="text-gray-700 font-medium mb-4">已上传图片</h3>
            <div class="space-y-3 max-h-[500px] overflow-y-auto pr-2">
              <div
                v-for="(image, index) in images"
                :key="index"
                @click="selectedImageIndex = index"
                class="flex items-center p-3 rounded-lg transition-all cursor-pointer"
                :class="{
                  'bg-blue-50 ring-2 ring-blue-500': selectedImageIndex === index,
                  'hover:bg-gray-50': selectedImageIndex !== index,
                }">
                <img :src="image.original" class="w-16 h-16 object-cover rounded-lg" />
                <div class="ml-4 flex-1">
                  <p class="text-sm font-medium text-gray-900 truncate">{{ image.name }}</p>
                  <p class="text-xs text-gray-500 mt-1">
                    {{ image.processed ? "处理完成" : "处理中..." }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: Selected Image Preview -->
        <div class="flex-1 bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
          <div v-if="selectedImageIndex !== null && images[selectedImageIndex]" class="space-y-6">
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-medium text-gray-900">{{ images[selectedImageIndex].name }}</h3>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="space-y-2">
                <p class="text-sm font-medium text-gray-600">原图</p>
                <div class="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  <img :src="images[selectedImageIndex].original" class="w-full h-full object-contain" alt="Original image" />
                </div>
              </div>
              <div class="space-y-2">
                <p class="text-sm font-medium text-gray-600">处理后</p>
                <div class="relative aspect-square bg-[url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAAYdEVYdFRpdGxlAENhbnZhcyBCYWNrZ3JvdW5kTPrL+wAAABd0RVh0QXV0aG9yAFBhYmxvIE1hcmNvcyBKLs3mAP8AAAAsdEVYdERlc2NyaXB0aW9uAEJhY2tncm91bmQgcGF0dGVybiBmb3IgY2FudmFzZXOBBFj6AAAAJ3RFWHRDcmVhdGlvbiBUaW1lAE1vbiBKdW4gMjAgMTY6MDA6NDYgMjAxMZkQAwsAAAATdEVYdFNvZnR3YXJlAEdOT01FIEljb27E/6eKAAAASHRFWHRDb3B5cmlnaHQAQ0MgQXR0cmlidXRpb24tTm9uQ29tbWVyY2lhbC1TaGFyZUFsaWtlIGh0dHA6Ly9jcmVhdGl2ZWNvbW1vbnMub3JnL7fvWYsAAAB6SURBVDiNY/z//z8DJYCJIt3UNYCFgYGBgfFW8hpGBgYGvYu5j+TIMYAFmyQjIyMDAwMDQ8GVvAcgGsUQdAOQAUjNegy7MAxABugGwQwh2gB0gGwIMQagA2RDCBqAC8AMwWsAIQA3BK8BhADYEKwGEANghqAYQA4AAKuXJBAXHHkzAAAAAElFTkSuQmCC')] rounded-lg overflow-hidden">
                  <img v-if="images[selectedImageIndex].processed" :src="images[selectedImageIndex].processed" class="w-full h-full object-contain" alt="Processed image" />
                  <div v-else class="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-sm">
                    <div class="text-center">
                      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                      <p class="text-gray-600">处理中...</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-else class="h-[600px] flex items-center justify-center">
            <div class="text-center">
              <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p class="mt-2 text-sm text-gray-500">请选择要查看的图片</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
