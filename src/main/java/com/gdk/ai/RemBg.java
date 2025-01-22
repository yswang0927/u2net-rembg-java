package com.gdk.ai;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class RemBg {

    static {
        // 加载 opencv 依赖的动态链接库
        System.load(RemBg.class.getClassLoader().getResource("libopencv_java4100.so").getFile());
    }

    static final float[] MEAN = { 0.485f, 0.456f, 0.406f };
    static final float[] STD = { 0.229f, 0.224f, 0.225f };

    /**
     * 移除背景。
     *
     * @param imagePath 图片路径
     * @return 移除结果
     * @throws Exception 失败异常
     */
    public static RemResult removeBackground(String imagePath) throws Exception {
        return removeBackgroundOrForeground(imagePath, false);
    }

    /**
     * 移除前景。
     *
     * @param imagePath 图片路径
     * @return 移除结果
     * @throws Exception 失败异常
     */
    public static RemResult removeForeground(String imagePath) throws Exception {
        return removeBackgroundOrForeground(imagePath, true);
    }

    /**
     * 移除背景或前景。
     *
     * @param imagePath 路径
     * @param removeForeground 是否是移除前景
     * @return 移除结果
     * @throws Exception 失败异常
     */
    private static RemResult removeBackgroundOrForeground(String imagePath, boolean removeForeground) throws Exception {
        //模型下载地址：https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
        OrtSession session;
        try (InputStream is = RemBg.class.getClassLoader().getResourceAsStream("u2net.onnx")) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[8192]; // 可以调整缓冲区大小
            while ((nRead = is.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] modelBytes = buffer.toByteArray();

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            session = env.createSession(modelBytes, sessionOptions);
            // 解析模型输入参数元数据信息
            final String inputName = session.getInputNames().iterator().next();
            Map<String, NodeInfo> inputInfo = session.getInputInfo();
            NodeInfo inputNodeInfo = inputInfo.get(inputName);
            TensorInfo inputTensorInfo = (TensorInfo) inputNodeInfo.getInfo();
            // eg: FLOAT, UINT8 等
            //OnnxJavaType inputType = inputTensorInfo.type;
            // eg: (1, 3, 320, 320)
            final long[] inputShape = inputTensorInfo.getShape();
            //final long inputNumElements = inputTensorInfo.getNumElements();
            // 模型要求的通道数、高、宽
            //final int channels = (int) inputShape[1];
            //final int netHeight = (int) inputShape[2];
            final int netWidth = (int) inputShape[3];

            final int modelSize = netWidth;

            // 解析模型输出参数元数据信息
            //final String outputName = session.getOutputNames().iterator().next();
            //Map<String, NodeInfo> outputInfo = session.getOutputInfo();
            //NodeInfo outputNodeInfo = outputInfo.get(outputName);
            //TensorInfo ouputTensorInfo = (TensorInfo) outputNodeInfo.getInfo();
            //OnnxJavaType outputType = ouputTensorInfo.type;
            //final long[] outputShape = ouputTensorInfo.getShape();
            //final long outputNumElements = ouputTensorInfo.getNumElements();

            // 使用 opencv 读取图像
            Mat srcImg = Imgcodecs.imread(imagePath);
            final int srcWidth = srcImg.cols();
            final int srcHeight = srcImg.rows();

            // 预处理图像
            // 1. 调整图像大小到模型输入要求的大小：等比缩放，不足的填充
            Mat resized = new Mat();
            final float ratio = 1.0f * modelSize / Math.max(srcHeight, srcWidth);
            final int newHeight = (int)(srcHeight * ratio);
            final int newWidth = (int)(srcWidth * ratio);
            Imgproc.resize(srcImg, resized, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_LANCZOS4);
            //Imgproc.resize(srcImg, resized, new Size(newWidth, newHeight));
            final int resizedWidth = resized.cols();
            final int resizedHeight = resized.rows();
            final boolean needFill = resizedWidth != modelSize || resizedHeight != modelSize;
            if (needFill) {
                Core.copyMakeBorder(resized, resized, 0, modelSize - newHeight, 0, modelSize - newWidth, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));
            }

            // 2. 转换 BGR -> RGB
            Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);

            // 3. 进行归一化、均值和方处理
            float[] normalizedData = normalizeAndPrepareInput(resized, modelSize);
            // 归一化: 确保图像的RGB值被正确归一化到[0, 1]或[-1, 1]范围
            //resized.convertTo(resized, CvType.CV_32FC1, 1.0 / 255.0f);
            // 均值和方差的处理
            //Core.meanStdDev(resized, new MatOfDouble(0.485, 0.456, 0.406), new MatOfDouble(0.229, 0.224, 0.225));

            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(normalizedData), inputShape);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, inputTensor);

            // 4. 执行推理
            try (OrtSession.Result results = session.run(inputs)) {
                OnnxTensor outputTensor = (OnnxTensor) results.get(0);
                float[][][][] outputData = (float[][][][]) outputTensor.getValue();
                float[][] maskData = outputData[0][0];

                // 5. 后处理推理数据，获得蒙版
                Mat outputMask = postProcessOutput(maskData, removeForeground);
                Imgproc.cvtColor(outputMask, outputMask, Imgproc.COLOR_RGB2BGR);
                // 6. 还原蒙版尺寸到原始图像大小
                if (needFill) {
                    Rect rect = new Rect(0, 0, resizedWidth, resizedHeight);
                    outputMask = outputMask.submat(rect);
                }

                Mat resultMask = new Mat();
                //Imgproc.resize(outputMask, resultMask, new Size(srcWidth, srcHeight), 0, 0, Imgproc.INTER_LANCZOS4);
                Imgproc.resize(outputMask, resultMask, new Size(srcWidth, srcHeight));
                if (resultMask.type() != CvType.CV_8UC1) {
                    Mat mask8u = new Mat();
                    resultMask.convertTo(mask8u, CvType.CV_8UC1);
                    resultMask = mask8u;
                }

                // 创建一个支持透明度的四通道输出图像
                Mat resultImage = new Mat(srcHeight, srcWidth, CvType.CV_8UC4);
                // 遍历图像的每个像素
                for (int y = 0; y < srcHeight; y++) {
                    for (int x = 0; x < srcWidth; x++) {
                        double[] originalPixel = srcImg.get(y, x);
                        double[] maskPixel = resultMask.get(y, x);
                        // 假设蒙版中白色区域为前景，黑色区域为背景
                        // 白色为 255，黑色为 0
                        double alpha = maskPixel[0] / 255.0;
                        // 设置输出图像的像素
                        double[] resultPixel = new double[] {
                                originalPixel[0],
                                originalPixel[1],
                                originalPixel[2],
                                alpha * 255
                        };
                        resultImage.put(y, x, resultPixel);
                    }
                }

                //Mat result = new Mat();
                //Core.bitwise_and(srcImg, resizedMask, result);
                // 可以使用 applyMaskWithBgColor() 改变背景颜色
                //result = applyMaskWithBgColor(result, resizedMask, new Scalar(255,255,0));

                return new RemResult(resultMask, resultImage);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }
    }

    private static float[] normalizeAndPrepareInput(Mat img, int inputSize) {
        float[] data = new float[inputSize * inputSize * 3];
        int index = 0;
        // 进行归一化、均值和方处理
        for (int y = 0; y < inputSize; ++y) {
            for (int x = 0; x < inputSize; ++x) {
                double[] pixel = img.get(y, x);
                for (int c = 0; c < 3; ++c) {
                    data[index++] = ((float) pixel[c] / 255.0f - MEAN[c]) / STD[c];
                }
            }
        }

        // 调整图片中的HWC [高度,宽度,通道] -> CHW[通道,高度,宽度]
        float[] transposedData = new float[data.length];
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < inputSize * inputSize; ++i) {
                transposedData[c * inputSize * inputSize + i] = data[i * 3 + c];
            }
        }
        return transposedData;
    }

    private static Mat postProcessOutput(float[][] outputData, boolean removeForeground) {
        final int height = outputData.length;
        final int width = outputData[0].length;

        // Create a Mat to hold the processed output
        Mat result = Mat.zeros(height, width, CvType.CV_32FC1);

        // Apply transformations: invert, normalize, scale to [0, 255]
        float min_value = Float.MAX_VALUE;
        float max_value = -Float.MAX_VALUE;

        // First pass to find min and max
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = outputData[y][x];
                if (value < min_value) {
                    min_value = value;
                }
                if (value > max_value) {
                    max_value = value;
                }
            }
        }

        // Second pass to apply transformations
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = outputData[y][x];
                // 黑白色翻转，如果翻转表示去除前景，否则是去除背景
                if (removeForeground) {
                    value = 1 - value; // Invert
                }
                value = (value - min_value) / (max_value - min_value); // Normalize
                value *= 255; // Scale to [0, 255]
                outputData[y][x] = value;
                result.put(y, x, value);
            }
        }

        return result;
    }

    public static Mat applyMaskWithBgColor(Mat src, Mat mask, Scalar bgColor) {
        Mat dst = src.clone();
        // Only take the first three values from bgColor.val if it's a 3-channel image
        double[] colorValues = new double[src.channels()];
        for (int c = 0; c < src.channels(); c++) {
            colorValues[c] = bgColor.val[c];
        }

        // Iterate over all pixels in the mask and set the corresponding pixel in dst to the background color
        for (int y = 0; y < mask.rows(); y++) {
            for (int x = 0; x < mask.cols(); x++) {
                if (mask.get(y, x)[0] == 0) { // If mask pixel is zero, apply background color
                    dst.put(y, x, colorValues); // Set background color
                }
            }
        }
        return dst;
    }

}
