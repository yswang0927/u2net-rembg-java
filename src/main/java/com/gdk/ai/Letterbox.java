package com.gdk.ai;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * 用于缩放图片
 */
public class Letterbox {

    private Size newShape ;
    private final double[] color = new double[] {114, 114, 114};
    private final Boolean auto = false;
    private final Boolean scaleUp = true;
    private Integer stride = 32;

    private double ratio;
    private double dw;
    private double dh;

    public Letterbox(int w, int h) {
        this.newShape = new Size(w, h);
    }

    public Letterbox() {
        this(640, 640);
    }

    public double getRatio() {
        return ratio;
    }

    public double getDw() {
        return dw;
    }

    public double getDh() {
        return dh;
    }

    public Integer getWidth() {
        return (int) this.newShape.width;
    }

    public Integer getHeight() {
        return (int) this.newShape.height;
    }

    public void setNewShape(Size newShape) {
        this.newShape = newShape;
    }

    public void setStride(Integer stride) {
        this.stride = stride;
    }

    /**
     * 调整图像大小和填充图像，使满足步长约束，并记录参数
     */
    public Mat letterbox(Mat im) {
        // 当前形状 [height, width]
        int[] shape = { im.rows(), im.cols() };
        // Scale ratio (new / old)
        double r = Math.min(this.newShape.height / shape[0], this.newShape.width / shape[1]);
        // 仅缩小，不扩大（为了mAP）
        if (!this.scaleUp) {
            r = Math.min(r, 1.0);
        }

        // Compute padding
        Size newUnpad = new Size(Math.round(shape[1] * r), Math.round(shape[0] * r));
        // 计算距离目标尺寸的padding像素数
        double dw = this.newShape.width - newUnpad.width;
        double dh = this.newShape.height - newUnpad.height;
        if (this.auto) { // 最小矩形
            dw = dw % this.stride;
            dh = dh % this.stride;
        }

        // 填充的时候两边都填充一半，使图像居于中心
        dw /= 2.0f;
        dh /= 2.0f;

        // 等比缩放
        if (shape[1] != newUnpad.width || shape[0] != newUnpad.height) {
            Imgproc.resize(im, im, newUnpad, 0, 0, Imgproc.INTER_LINEAR);
        }

        // 图像四周padding填充，至此原图与目标尺寸一致
        int top = (int) Math.round(dh - 0.1f);
        int bottom = (int) Math.round(dh + 0.1f);
        int left = (int) Math.round(dw - 0.1f);
        int right = (int) Math.round(dw + 0.1f);
        // 将图像填充为正方形
        Core.copyMakeBorder(im, im, top, bottom, left, right, Core.BORDER_CONSTANT, new org.opencv.core.Scalar(this.color));
        this.ratio = r;
        this.dh = dh;
        this.dw = dw;

        return im;
    }

}
