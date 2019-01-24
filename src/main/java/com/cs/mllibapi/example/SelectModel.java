package com.cs.mllibapi.example;

import com.cs.mllibapi.evaluation.RankingEvaluator;
import com.cs.mllibapi.rankmodel.GBDTAndLRRankModel;
import com.cs.mllibapi.rankmodel.LRRankModel;
import com.cs.mllibapi.rankmodel.RankModel;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

// 选择运行排序模型
// spark-submit --driver-memory 128G --executor-memory 128G --conf spark.driver.maxResultSize=128g  --master localhost[4] --class "com.cs.mllibapi.example.SelectModel"  CTRmodel-1.0-SNAPSHOT.jar 1
public final class SelectModel {

    //输入特征
    private final static String feature_save_file = "data/dataprocess/result/features";

    public static void main(String[] args) throws Exception {
        String selected;
        try {
            selected = args[0];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("未选择模型，下面使用默认的模型");
            selected = "3";
        }

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

        System.out.println("\n训练集数量=" + training.count() + ",测试集数量=" + test.count());
        System.out.println("输入特征（样例）：");
        training.take(3).forEach(s -> System.out.println(s));


        RankModel rankModel = new GBDTAndLRRankModel();    //排序模型接口

        //选择模型
        switch (selected) {
            case "1":
                rankModel = new LRRankModel();
                break;
            case "2":
                rankModel = new GBDTAndLRRankModel();
                break;
        }
        System.out.println("\n使用模型：" + rankModel.getClass().getSimpleName());

        System.out.println("模型参数：");
        System.out.println(rankModel.getParams());

        long startTime = System.currentTimeMillis();

        //训练
        rankModel.train(training);

        //预测测试集的label
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = rankModel.transform(test);
        predictionAndLabels.cache();

        System.out.println("\n预测值 <-> 真实值：");
        predictionAndLabels.take(30).forEach(s -> {
            System.out.println(s._1 + " <-> " + s._2);
        });

        // 评测
        RankingEvaluator rankingEvaluator = new RankingEvaluator();
        rankingEvaluator.evaluate(rankModel.getClass().getSimpleName(), predictionAndLabels);

        long endTime = System.currentTimeMillis();
        long usedTime = (endTime - startTime) / 1000;
        System.out.println("总耗时 = " + usedTime + " s");

        jsc.close();
    }
}
