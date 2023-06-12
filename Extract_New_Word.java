package com.example.BossZhipin.NLP;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import com.sun.jna.Native;

public class Demo {
 private static final boolean False = false;
static Connection conn;
 static PreparedStatement st;
 static ResultSet rs;

 static File file=new File("C:\\Users\\86198\\Desktop\\学校\\大三\\专利信息检索与分析\\keyword_extraction-master\\new_word.txt");

  public static void main(String[] args) throws Exception {
      //初始化
      CLibrary instance = (CLibrary)Native.loadLibrary("C:\\Program Files (x86)\\jsoup\\jsoup-1.11.2-javadoc\\demo\\src\\main\\java\\com\\example\\BossZhipin\\NLP\\win64\\NLPIR", CLibrary.class);
      int init_flag = instance.NLPIR_Init("C:\\Program Files (x86)\\jsoup\\jsoup-1.11.2-javadoc\\demo\\src\\main\\java\\com\\example\\BossZhipin\\NLP", 1, "0");
      String resultString = null;
      if (0 == init_flag) {
          resultString = instance.NLPIR_GetLastErrorMsg();
          System.err.println("初始化失败！\n"+resultString);
          return;
      }

      BufferedWriter br=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file),"UTF-8"));
      BufferedReader readme = new BufferedReader(new InputStreamReader(new FileInputStream("C:\\Users\\86198\\Desktop\\学校\\大三\\专利信息检索与分析\\new_recognization.txt"), "UTF-8"));
      String line;
      while((line = readme.readLine()) != null){
        try{
          // System.out.print(line);
          String newWord=instance.NLPIR_GetNewWords(line,10,False);
          // System.out.println(newWord);
          br.write(newWord.replace("#", "\n"));
          br.flush();
        } catch (Exception e) {
            System.out.println("错误信息：");
            e.printStackTrace();
          }
      }
     /*
      s（str） - 要处理的中文文本。
      max_words（int） - 要返回的新单词的最大数量。
      weighted（bool） - 是否返回新单词的权重。
    */
    instance.NLPIR_Exit();
    br.close();
    readme.close();
    }
}
