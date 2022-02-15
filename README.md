### 1. 產生.h5 model

* 打開資料夾進入**Python_validation**

  ```shell
  cd Python_validation
  ```

* 用conda搭建已經做好的環境(名稱為test_env)

  將 ``environment.yml``的最後一行

  ``prefix: /home/kung/anaconda3/envs/test_env`` 中的 \<kung\> 改成自己的 username 

  ``` shell
  conda env create -f environment.yml
  ```

* 啟用環境

  ``` 
  conda activate test_env
  ```

* 啟用jupyter-lab  (port=8989 可以取自己喜歡的port)

  ```
  jupyter-lab --ip=0.0.0.0 --port=8989
  ```

* 點選左側 **Main.ipynb**

  ![img01](./readme_imgs/img01.png)

* 將 **Main.ipynb** 全部執行完畢產生一個**model01.h**檔案

* 將**model01.h**移動到**data1/model01**中

* 將整個**data1**資料夾移動到**\~/data1**中(如果沒有data1請先建立)，等等要跟docker做共享

* 完成後**\~/data1**的資料如下
  ``` shell
  (base) kung@kung-ntu103:~/data1$ tree
  .
  ├── batch_input_params.json
  ├── input_params.json
  └── model01
      ├── images
      │   ├── birdman.bmp
      │   ├── img01.bmp
      │   ├── img02.bmp
      │   ├── img03.bmp
      │   ├── img04.bmp
      │   ├── img05.bmp
      │   ├── img06.bmp
      │   ├── img07.bmp
      │   ├── img08.bmp
      │   ├── img09.bmp
      │   ├── img10.bmp
      │   └── img11.bmp
      └── model01.h5
  ```
### 2. 用KL520 toolchain 產生可以.nef model檔

* pull docker

  ``` shell
  docker pull kneron/toolchain:v0.14.1
  ```

* 將本機的的 ~/data1 資料夾 map 到 docker toolchain520中的 /data1 將docker run 起來 關掉就直接刪除container (\<kung>改成自己的使用者名稱)

  ``` shell
  docker run --rm -it -v /home/kung/data1:/data1 kneron/toolchain:v0.14.1
  ```

* 此步驟之後都在docker中操作

* keras h5 轉成 onnx

  ``` shell
  python /workspace/scripts/convert_model.py keras /data1/model01/model01.h5 /data1/model01/model01.h5.onnx
  ```

* 此時**data1**中檔案結構如下

  ``` shell
  .
  ├── batch_input_params.json
  ├── input_params.json
  └── model01
      ├── images
      │   ├── birdman.bmp
      │   ├── img01.bmp
      │   ├── img02.bmp
      │   ├── img03.bmp
      │   ├── img04.bmp
      │   ├── img05.bmp
      │   ├── img06.bmp
      │   ├── img07.bmp
      │   ├── img08.bmp
      │   ├── img09.bmp
      │   ├── img10.bmp
      │   └── img11.bmp
      ├── model01.h5
      └── model01.h5.onnx
  ```

* **fpAnalyserCompiler** (For KDP520)

  ``` shell
  python /workspace/scripts/fpAnalyserCompilerIpevaluator_520.py -t 8
  ```

  成功會出現

  ``` 
  [piano][warning][graph_gen.cc:85][GenerateGraph] Model [/data1/fpAnalyser/model01.h5.quan.wqbi.bie] is BIE, skip optimization config
  ```

* 此時**data1**中檔案結構如下

  ``` shell
  .
  ├── batch_input_params.json
  ├── compiler
  │   ├── command.bin
  │   ├── ioinfo.csv
  │   ├── ip_eval_prof.txt
  │   ├── setup.bin
  │   └── weight.bin
  ├── fpAnalyser
  │   └── model01.h5.quan.wqbi.bie
  ├── input_params.json
  └── model01
      ├── images
      │   ├── birdman.bmp
      │   ├   ...
      │   ├── img10.bmp
      │   └── img11.bmp
      ├── model01.h5
      └── model01.h5.onnx
  ```

* Compiler and Evaluator

  ``` shell
  cd /workspace/scripts && ./compilerIpevaluator_520.sh /data1/model01/model01.h5.onnx
  ```

  成功出現以下

  ``` 
  running compiler and IP evaluator...
  Compiler config generated.
  Compilation and IP Evaluation finished.
  ```

* Batch-Compile
  
  ``` shell
  python /workspace/scripts/batchCompile_520.py
  ```
  
  成功出現以下訊息 (綠色字體)
  
  ``` 
  [tool][info][batch_compile.cc:701][VerifyOutput]      addr: 0x60345b30, size: 0xa0
  [tool][info][batch_compile.cc:704][VerifyOutput] 
  
  [tool][info][batch_compile.cc:708][VerifyOutput]   end addr 0x60345bd0, 
  [tool][info][batch_compile.cc:710][VerifyOutput] total bin size 0x2abae0
  ```
  
* 此時檔案結構如下

  ``` shell
  .
  ├── batch_compile
  │   ├── all_models.bin
  │   ├── batch_compile_bconfig.json
  │   ├── batch_compile_config.json
  │   ├── batch_compile.log
  │   ├── compile.log
  │   ├── fw_info.bin
  │   ├── fw_info.txt
  │   ├── model01_config.json
  │   ├── model01_modelid_1_command.bin
  │   ├── model01_modelid_1_ioinfo.csv
  │   ├── model01_modelid_1_setup.bin
  │   ├── model01_modelid_1_weight.bin
  │   └── models_520.nef
  ├── batch_input_params.json
  ├── compiler
  │   ├── command.bin
  │   ├── ioinfo.csv
  │   ├── ip_eval_prof.txt
  │   ├── setup.bin
  │   └── weight.bin
  ├── fpAnalyser
  │   └── model01.h5.quan.wqbi.bie
  ├── input_params.json
  └── model01
      ├── images
      │   ├── birdman.bmp
      │   ├──   ....
      │   └── img11.bmp
      ├── model01.h5
      └── model01.h5.onnx
  ```

* 關掉 KL520 toolchain 的**container**

  ``` shell
  exit
  ```

* **~/data1/batch_compile/models_520.nef** 就是我們要在kl520 上面跑得model

### 3. 在vm上用kl520 inference

* 到http://doc.kneron.com/docs/#520_1.4.0.0/getting_start/ 下載vm，然後用vmware打開

* 順便到 https://www.kneron.com/tw/support/developers/ 的**KNEO Stem (USB Dongle)**項目中，下載host_lib，這裡示範的版本為**host_lib_v1.0.0.zip**

  ![img02](/home/kung/Workspace/kl520_image_test/readme_imgs/img02.png)**

  

* 打開vmware，打開vm，密碼是 **Kneron**

* 在這裡我們將host_lib解壓縮到vm的桌面上，路徑為**~/Desktop/host_lib/**

* 將整個 **kl520_image_test** 資料夾拉進去VM的桌面上

* 在**kl520_image_test**資料夾中打開terminal

  輸入ls看到以下結構

  ``` shell
  Kneron@ubuntu:~/Desktop/kl520_image_test$ ls
  data1  images  KL_520_example  my_test_model  Python_validation  README.md
  ```

* 將我們自己做的**224*224**影像放入**host_lib**中

  ``` shell
  cp -r data1/model57/images/ ~/Desktop/host_lib/
  ```

* 將我們自己compile的模型檔案(.nef)放入**host_lib**中

  ``` shell
  cp -r my_test_model/ ~/Desktop/host_lib/input_models/KL520/
  ```

* 將自己寫的**kl520_dme_image_inference_example**放入**host_lib**中

  ``` shell
  cd KL_520_example/
  cp -r * ~/Desktop/host_lib/example/KL520/
  ```

* 一切準備完成，開始**build**

* 移動到**host_lib**中，建立build

  ``` shell
   cd ~/Desktop/host_lib/
   mkdir build && cd build
  ```

* cmake，這裡需要啟用Opencv範例

  ``` shell
  cmake -DBUILD_OPENCV_EX=on ..
  ```

  完成顯示

  ```
  -- Build for UNIX environment
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /home/Kneron/Desktop/host_lib/build
  ```

* make

  ``` shell
  make -j4
  ```

* 確定**KL520**有連接到VM

  ``` shell
  lsusb
  ```

  如果有連結到會出現以下結果

  ```
  Kneron@ubuntu:~/Desktop/host_lib/build/bin$ lsusb
  Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
  Bus 003 Device 004: ID 0e0f:0002 VMware, Inc. Virtual USB Hub
  Bus 003 Device 003: ID 0e0f:0002 VMware, Inc. Virtual USB Hub
  Bus 003 Device 005: ID 3231:0100  
  Bus 003 Device 002: ID 0e0f:0003 VMware, Inc. Virtual Mouse
  Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
  Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
  Bus 002 Device 001: ID 1d6b:0001 Linux Foundation 1.1 root hub
  ```

* 移動到**bin**中執行範例

  ``` shell
  cd bin
  ./kl520_dme_image_inference_example
  ```

* 滑鼠點擊跳出的影像視窗，隨便鍵盤按一個按鍵，就可以結束**inference**

  結果如下

  ![result](https://raw.githubusercontent.com/kung-bill/kl520_image_test/master/result.png)

* 可以試試看不同圖片的結果，有些結果差異很大，有些很小
