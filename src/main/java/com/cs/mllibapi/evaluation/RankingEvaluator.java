package com.cs.mllibapi.evaluation;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import scala.Tuple2;
import scala.collection.mutable.StringBuilder;

//排序模型评测器
public class RankingEvaluator {

    //指标
    private double cross_entropy;
    private int tp, tn, fp, fn;
    private double mse;

    //计算指标
    private void calculate(JavaRDD<Tuple2<Object, Object>> predictionAndLabels) {
        //交叉熵
        cross_entropy = predictionAndLabels.map(tuple -> {
            Double predict = (Double) tuple._1;
            Double label = (Double) tuple._2;
            if (label.equals(1.0) && predict > 0) {
                return -Math.log(predict);
            } else {
                return 0.0;
            }
        }).reduce((a, b) -> a + b) / predictionAndLabels.count();

        //正负样本
        //label=1,预测>0.5
        tp = predictionAndLabels.map(tuple -> {
            Double predict = (Double) tuple._1;
            Double label = (Double) tuple._2;
            if (label.equals(1.0) && predict > 0.5)
                return 1;
            else
                return 0;
        }).reduce((a, b) -> a + b);

        //label=0,预测<0.5
        tn = predictionAndLabels.map(tuple -> {
            Double predict = (Double) tuple._1;
            Double label = (Double) tuple._2;
            if (label.equals(0.0) && predict < 0.5)
                return 1;
            else
                return 0;
        }).reduce((a, b) -> a + b);

        //label=0,预测>0.5
        fp = predictionAndLabels.map(tuple -> {
            Double predict = (Double) tuple._1;
            Double label = (Double) tuple._2;
            if (label.equals(0.0) && predict > 0.5)
                return 1;
            else
                return 0;
        }).reduce((a, b) -> a + b);

        //label=1,预测<0.5
        fn = predictionAndLabels.map(tuple -> {
            Double predict = (Double) tuple._1;
            Double label = (Double) tuple._2;
            if (label.equals(1.0) && predict < 0.5)
                return 1;
            else
                return 0;
        }).reduce((a, b) -> a + b);

        //均方差
        mse = predictionAndLabels.map(tuple -> {
            Double predict = (Double) tuple._1;
            Double label = (Double) tuple._2;
            return Math.pow((predict - label), 2);
        }).reduce((a, b) -> a + b) / predictionAndLabels.count();

    }

    /**
     * 评测排序模型，直接显示结果，并返回一个表示评测结果的字符串
     *
     * @param predictionAndLabels 预测值和真实值
     */
    public String evaluate(String modelName, JavaRDD<Tuple2<Object, Object>> predictionAndLabels) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(modelName).append("\n");
        stringBuilder.append("评测结果：").append("\n");

        // Get evaluation metrics
        // https://www.cnblogs.com/charlesblc/p/6252759.html
        BinaryClassificationMetrics binaryClassificationMetrics
                = new BinaryClassificationMetrics(predictionAndLabels.rdd());
        stringBuilder.append("area under ROC (AUC) = " + binaryClassificationMetrics.areaUnderROC()).append("\n");
        stringBuilder.append("area under PR = " + binaryClassificationMetrics.areaUnderPR()).append("\n");

        calculate(predictionAndLabels);

        //交叉熵
        stringBuilder.append("交叉熵 = " + cross_entropy).append("\n");   //越小越好

        //正负样本
        stringBuilder.append("测试集总数 = " + predictionAndLabels.count()).append("\n");
        stringBuilder.append("true positive = " + tp).append("\n");
        stringBuilder.append("true negitive = " + tn).append("\n");
        stringBuilder.append("false positive = " + fp).append("\n");
        stringBuilder.append("false negitive = " + fn).append("\n");

        //均方差
        stringBuilder.append("MSE = " + mse);

        System.out.println(stringBuilder.toString());
        return stringBuilder.toString();
    }


}



