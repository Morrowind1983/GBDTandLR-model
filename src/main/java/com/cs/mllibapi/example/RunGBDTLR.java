package com.cs.mllibapi.example;

import com.cs.mllibapi.evaluation.RankingEvaluator;
import com.cs.mllibapi.rankmodel.GBDTAndLRRankModel;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.HashMap;


//运行GBDT+LR模型，需设定参数
//spark-submit --driver-memory 128G --executor-memory 128G --conf spark.driver.maxResultSize=128g  --master localhost[8] --class "com.cs.mllibapi.example.RunGBDTLR"  CTRmodel-1.0-SNAPSHOT.jar 5 8
public final class RunGBDTLR {

    //输入特征
    private final static String feature_save_file = "data/dataprocess/result/features";

    public static void main(String[] args) throws Exception {
        //设定参数
        HashMap<String, Object> gbdtlr_params = new HashMap<>();
        try {
            gbdtlr_params.put("gbdt_maxdepth", Integer.parseInt(args[0]));
            gbdtlr_params.put("gbdt_numtrees", Integer.parseInt(args[1]));
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("参数错误");
            return;
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

        //模型
        GBDTAndLRRankModel rankModel = new GBDTAndLRRankModel();
        System.out.println("\n使用模型：" + rankModel.getClass().getSimpleName());

        rankModel.setParams(gbdtlr_params);
        System.out.println("使用参数：");
        System.out.println(rankModel.getParams());

        //训练
        rankModel.train(training);

        //预测测试集的label
        System.out.println("开始预测训练集的label：");
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = rankModel.transform(test);
        predictionAndLabels.cache();

        System.out.println("\n预测值 <-> 真实值：");
        predictionAndLabels.take(30).forEach(s -> {
            System.out.println(s._1 + " <-> " + s._2);
        });


        // 评测
        RankingEvaluator rankingEvaluator = new RankingEvaluator();
        rankingEvaluator.evaluate(rankModel.getClass().getSimpleName(), predictionAndLabels);


        jsc.close();
    }

}
