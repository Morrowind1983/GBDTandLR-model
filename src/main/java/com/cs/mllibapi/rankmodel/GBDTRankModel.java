package com.cs.mllibapi.rankmodel;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

//todo 暂不能用，由于spark api受限
public class GBDTRankModel implements RankModel {

    private GradientBoostedTreesModel gradientBoostedTreesModel;

    /**
     * 训练
     *
     * @param training 数据是已经处理好的，执行train()时，不会对label和特征进行变换
     */
    @Override
    public void train(JavaRDD<LabeledPoint> training) {

        // The defaultParams for Regression use SquaredError by default.
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Regression");
        //boostingStrategy.setNumIterations(3);
        //boostingStrategy.getTreeStrategy().setMaxDepth(5);

        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        gradientBoostedTreesModel = GradientBoostedTrees.train(training, boostingStrategy);
    }

    /**
     * 执行预测
     *
     * @param samples 输入特征，LabeledPoints
     * @return 预测值和真实值
     */
    @Override
    public JavaRDD<Tuple2<Object, Object>> transform(JavaRDD<LabeledPoint> samples) {
        //todo 输出的 predict Label 是01值，不是概率
        return samples.map(point -> {
            Double prediction = gradientBoostedTreesModel.predict(point.features());
            return new Tuple2<>(prediction, point.label());
        });
    }

    @Override
    public String getParams() {
        return null;
    }

    @Override
    public void setParams(HashMap<String, Object> paramSet) {

    }
}
