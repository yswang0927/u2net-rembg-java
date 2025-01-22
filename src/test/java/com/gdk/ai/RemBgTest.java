package com.gdk.ai;

import java.io.File;

public class RemBgTest {
    public static void main(String[] args) throws Exception {
        File imgFile = new File("imgs/img2.jpg");
        RemBg.removeBg(imgFile.getAbsolutePath());
    }
}
