package com.gdk.ai;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class RemResult {
    private final Mat maskMat;
    private final Mat resultMat;

    public RemResult(Mat maskMat, Mat resultMat) {
        this.maskMat = maskMat;
        this.resultMat = resultMat;
    }

    public Mat getMaskImageMat() {
        return this.maskMat;
    }

    public Mat getResultImageMat() {
        return this.resultMat;
    }

    public void writeMaskImage(String destMaskImagePath) {
        Imgcodecs.imwrite(destMaskImagePath, this.maskMat);
    }

    public void writeResultImage(String destResultImagePath) {
        Imgcodecs.imwrite(destResultImagePath, this.resultMat);
    }

}
