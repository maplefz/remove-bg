import { AutoModel, AutoProcessor, RawImage } from "@huggingface/transformers";

let model = null;
let processor = null;
let isProcessing = false;
let processingQueue = [];

async function initializeModel() {
  try {
    self.postMessage({ type: 'loadingProgress', progress: 0, stage: '正在加载模型...' });

    model = await AutoModel.from_pretrained("briaai/RMBG-2.0", {
      config: { model_type: "custom" },
      device: "webgpu",
      progress_callback: (progress) => {
        console.log('progress', progress);
        self.postMessage({
          type: 'loadingProgress',
          progress: (progress.progress || 0).toFixed(2),
          stage: '正在加载模型...'
        });
      }
    });

    self.postMessage({ type: 'loadingProgress', progress: 50, stage: '正在加载处理器...' });

    processor = await AutoProcessor.from_pretrained("briaai/RMBG-2.0", {
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
      progress_callback: (progress) => {
        console.log('progress', progress);
        self.postMessage({
          type: 'loadingProgress',
          progress: (progress.progress || 0).toFixed(2),
          stage: '正在加载处理器...'
        });
      }
    });

    self.postMessage({ type: 'loadingProgress', progress: 100, stage: '加载完成' });
    self.postMessage({ type: 'modelLoaded' });
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: `Model initialization failed: ${error.message}`
    });
  }
}

async function processNextInQueue() {
  if (processingQueue.length === 0 || isProcessing) {
    return;
  }

  isProcessing = true;
  const { image, index } = processingQueue.shift();

  try {
    if (!model || !processor) {
      await initializeModel();
    }

    const imageBlob = await RawImage.fromBlob(image);
    const { pixel_values } = await processor(imageBlob);
    const { output } = await model({ input: pixel_values });

    // Generate final image
    const mask = await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(imageBlob.width, imageBlob.height);
    const canvas = new OffscreenCanvas(imageBlob.width, imageBlob.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageBlob.toCanvas(), 0, 0);
    const pixelData = ctx.getImageData(0, 0, imageBlob.width, imageBlob.height);

    for (let i = 0; i < mask.data.length; ++i) {
      pixelData.data[i * 4 + 3] = mask.data[i];
    }
    ctx.putImageData(pixelData, 0, 0);

    const blob = await canvas.convertToBlob();
    const processedImageUrl = URL.createObjectURL(blob);

    self.postMessage({
      type: 'processComplete',
      result: {
        processedImage: processedImageUrl,
        index: index
      }
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error.message
    });

    // If we get a session error, try to reinitialize the model
    if (error.message.includes('Session')) {
      model = null;
      processor = null;
      await initializeModel();
    }
  } finally {
    isProcessing = false;
    // Process next item in queue if any
    if (processingQueue.length > 0) {
      processNextInQueue();
    }
  }
}

self.addEventListener('message', async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case 'init':
      await initializeModel();
      break;
    case 'process':
      processingQueue.push({
        image: data.image,
        index: data.index
      });
      processNextInQueue();
      break;
  }
});
