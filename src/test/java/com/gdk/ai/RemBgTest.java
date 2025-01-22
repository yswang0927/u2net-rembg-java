package com.gdk.ai;

import java.io.File;

public class RemBgTest {
    public static void main(String[] args) throws Exception {

        File imgFile = new File("imgs/img7.jpg");

        // 删除背景
        RemResult remResult = RemBg.removeBackground(imgFile.getAbsolutePath());
        remResult.writeMaskImage("output-mask.png");
        remResult.writeResultImage("output.png");

        // 删除前景
        remResult = RemBg.removeForeground(imgFile.getAbsolutePath());
        remResult.writeMaskImage("output2-mask.png");
        remResult.writeResultImage("output2.png");

    }

}
