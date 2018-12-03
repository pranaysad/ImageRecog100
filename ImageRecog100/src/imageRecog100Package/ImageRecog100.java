/**
 * 
 */
package imageRecog100Package;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;


//import org.deeplearning4j.ui.api.UIServer;



/**
 * @author Pranay
 *
 */
public class ImageRecog100 {

	static void machinelearning()
    {		
		
		int height = 28; 		// The number of rows of a matrix. - Image is 28x28 pixels
		int width = 28; 		// The number of columns of a matrix. - Image is 28x28 pixels
		
		//int channels = 3;		// 	Grayscale implies single channel
		int channels = 1;		// 	Grayscale implies single channel
		int rngseed = 123;		// 	This random-number generator applies a seed to ensure that 
		  						//	the same initial weights are used when training. 
		  						//	 We’ll explain why this matters later.
		
		Random randNumGen = new Random(rngseed);
		
		int numEpochs = 1; 	// An epoch is a complete pass through a given dataset.
		
		File trainData = new File(ClassLoader.getSystemResource("training").getPath());
		System.out.println(trainData.getPath());
		
		File testData = new File(ClassLoader.getSystemResource("testing").getPath());
		System.out.println(testData.getPath());
		
		FileSplit train = new FileSplit(trainData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
		FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
		
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
		
		try {
			recordReader.initialize(train);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int batchSize = 10;		// How many examples to fetch with each step.
		int outputNum = 10; 	// Number of possible outcomes (e.g. labels 0 through 7).
		
		//	Exception in thread "ADSI prefetch thread" java.lang.RuntimeException: java.lang.UnsupportedOperationException: 
		//	Cannot do conversion to one hot using batched reader: 1 output classes
		//	but array.size(1) is 7 (must be equal to 1 or numClasses = 1)
		
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
		
		// Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(rngseed)
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .iterations(1)
	            .learningRate(0.001)
	            //.updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).momentum(0.9)
	            .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS)
	            .regularization(true).l2(1e-4)
	            .list()
	            .layer(0, new DenseLayer.Builder()
	                    .nIn(height * width * channels)
	                    .nOut(100)
	                    .activation(Activation.RELU)
	                    .weightInit(WeightInit.XAVIER)
	                    .build())
	            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                    .nIn(100)
	                    .nOut(outputNum)
	                    .activation(Activation.SOFTMAX)
	                    .weightInit(WeightInit.XAVIER)
	                    .build())
	            .pretrain(false).backprop(true)
	            .setInputType(InputType.convolutional(height,width,channels))
	            .build();
		
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	    
	    for(int i = 0; i<numEpochs; i++){
	    	System.out.println("Training Attempt " + i);
	        model.fit(dataIter);
	    }
	    
	    //	Channels (3) x rows x columns x batch size = 3 x 1024 x 1024 =  3145728
	    //	1048576 = 1024 x 1024
	    
	    //	Exception in thread "main" org.deeplearning4j.exception.DL4JInvalidInputException
	    //	Input size (3145728 columns; shape = [10, 3145728]) is invalid: 
	    //	does not match layer input size (layer # inputs = 1048576) 
	    //	(layer name: layer0, layer index: 0)
	    
	    System.out.println("**************************");
        System.out.println("***** EVALUATE MODEL *****");
        System.out.println("**************************");
        //Note: Training has 60K files and Testing has 10K files
        
        recordReader.reset();
	    
        try {
			recordReader.initialize(test);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
	    DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
	    // org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator.RecordReaderDataSetIterator(
	    //	RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels)
	    
	    scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
	    
	    Evaluation eval = new Evaluation(outputNum);
	    
		while(testIter.hasNext()){
	        
			DataSet next = testIter.next();
	        
			INDArray output = model.output(next.getFeatureMatrix());
	        
	        //System.out.println(next.getLabels().toString());
	        
	        eval.eval(next.getLabels(),output);
	    }
	    
	    System.out.println(eval.stats());
		
	    System.out.println(eval.confusionToString());
	    
	    //Initialize the user interface backend
	    //UIServer uiServer = UIServer.getInstance();
	    //uiServer.enableRemoteListener();
	    
		
		
	    
		System.out.println("machinelearning function END");
		
		
    }	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		System.out.println("ImageRecog100");
		
		machinelearning();
		
		
		
	}

}
