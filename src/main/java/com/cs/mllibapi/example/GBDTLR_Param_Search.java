package com.cs.mllibapi.example;

import com.cs.mllibapi.evaluation.RankingEvaluator;
import com.cs.mllibapi.rankmodel.GBDTAndLRRankModel;
import lombok.AllArgsConstructor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

//搜索gbdt+lr模型的参数
//spark-submit --driver-memory 128G --executor-memory 128G --conf spark.driver.maxResultSize=128g  --master localhost[8] --class "com.cs.mllibapi.example.GBDTLR_Param_Search"  CTRmodel-1.0-SNAPSHOT.jar
public final class GBDTLR_Param_Search {

    private final static String feature_save_file = "data/dataprocess/result/features";
    private final static String evaluation_log_file = "gbdtlr_log.txt";

    public static void main(String[] args) throws Exception {
        Logger logger = Logger.getLogger("org");
        logger.setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("app").setMaster("local[4]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // 加载数据，来自data_process.FeatureEngineer
        JavaRDD<LabeledPoint> data = jsc.objectFile(feature_save_file);

        // 划分训练、测试数据
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 12345);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1].cache();

        //todo 修改需要执行的参数
        List<ParmTuple> parmTupleList = new ArrayList<>();
//        parmTupleList.add(new ParmTuple(2, 2));
//        parmTupleList.add(new ParmTuple(3, 3));
//        parmTupleList.add(new ParmTuple(5, 5));
//        parmTupleList.add(new ParmTuple(5, 8));
//        parmTupleList.add(new ParmTuple(8, 5));
//        parmTupleList.add(new ParmTuple(8, 8));
//        parmTupleList.add(new ParmTuple(3, 10));
//        parmTupleList.add(new ParmTuple(10, 3));
        parmTupleList.add(new ParmTuple(12, 3));
        parmTupleList.add(new ParmTuple(15, 3));
        parmTupleList.add(new ParmTuple(10, 5));
        parmTupleList.add(new ParmTuple(12, 5));
        parmTupleList.add(new ParmTuple(15, 5));

        for (ParmTuple parmTuple : parmTupleList) {
            long startTime = System.currentTimeMillis();

            //模型
            GBDTAndLRRankModel rankModel = new GBDTAndLRRankModel();
            System.out.println("\n使用模型：" + rankModel.getClass().getSimpleName());

            //设定参数
            HashMap<String, Object> paramMap = new HashMap<>();
            paramMap.put("gbdt_maxdepth", parmTuple.gbdt_maxdepth);
            paramMap.put("gbdt_numtrees", parmTuple.gbdt_numtrees);
            rankModel.setParams(paramMap);

            System.out.println("使用参数：");
            System.out.println(rankModel.getParams());

            //训练
            rankModel.train(training);

            //预测测试集的label
            System.out.println("开始预测训练集的label：");
            JavaRDD<Tuple2<Object, Object>> predictionAndLabels = rankModel.transform(test);
            predictionAndLabels.cache();

            // 评测
            RankingEvaluator rankingEvaluator = new RankingEvaluator();
            String log = rankingEvaluator.evaluate("GBDT+LR", predictionAndLabels);

            long endTime = System.currentTimeMillis();
            long usedTime = (endTime - startTime) / 1000;
            System.out.println("总耗时 = " + usedTime + " s");

            //如果文件存在，则追加内容；如果文件不存在，则创建文件
            FileWriter fw = null;
            try {
                File f = new File(evaluation_log_file);
                fw = new FileWriter(f, true);
            } catch (IOException e) {
                e.printStackTrace();
            }
            PrintWriter pw = new PrintWriter(fw);
            pw.println("==========================");
            pw.println(log + "\n");
            pw.println("参数：\n" + rankModel.getParams() + "\n");
            pw.println("总耗时 = " + usedTime + " s");
            pw.flush();
            try {
                fw.flush();
                pw.close();
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            predictionAndLabels.unpersist();
        }

        jsc.close();
    }

    @AllArgsConstructor
    private static class ParmTuple {
        public Integer gbdt_maxdepth;
        public Integer gbdt_numtrees;
    }
}
