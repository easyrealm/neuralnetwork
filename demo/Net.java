package neuronet;

import java.util.*;


public class Net
{
    private double beta;
    private double alpha;
    private double threshold;
    private int layerCount;
    private int[] layerConfig;
    private int iterCount;
    private Layer[] layers;
    private Layer inputLayer;
    private Layer outputLayer;
    private double[][] input;
    private double[][] output;
    public static Random random = new Random();
    public static int RAND_MAX = 32768;

    public Net(double beta, double alpha, double threshold, int layerCount, int[] layerConfig, int iterCount, double[][] input, double[][] output)
    {
        this.beta = beta;
        this.alpha = alpha;
        this.threshold = threshold;
        this.layerCount = layerCount;
        this.layerConfig = layerConfig;
        this.iterCount = iterCount;
        this.input = input;
        this.output = output;
        init();
    }

    public Layer getLayer(int index)
    {
        if((index >= 0)&&(index < layerCount))
            return layers[index];
        else return null;
    }

    private void init()
    {
        layers = new Layer[layerCount];
        layers[0] = new InputLayer(0, layerConfig[0], this);
        for(int i=1;i<layerCount-1;i++)
        {
            layers[i] = new Layer(i, layerConfig[i], this);
        }
        layers[layerCount - 1] = new OutputLayer(layerCount - 1, layerConfig[layerCount - 1], this);
        //assign the input and output layer
        inputLayer = (InputLayer) layers[0];
        outputLayer = (OutputLayer) layers[layerCount - 1];
    }

    private double mse(double[] tgt)
    {
        double sum = 0;
        int size = tgt.length;
        for(int i=0;i<size;i++)
        {
            sum += Math.pow(outputLayer.getNeurone(i).getOut() - tgt[i],2);
        }
        sum /= 2.0;
        return sum;
    }

    private void ffwd(double[] in)
    {
        ((InputLayer)inputLayer).forward(in);
        for(int i=1;i<layerCount;i++)
            layers[i].forward(layers[i-1]);
    }

    private void bkwd(double[] tgt)
    {
        ((OutputLayer)outputLayer).backward(tgt);
        for(int i=layerCount - 2;i>0;i--)
            layers[i].backward(layers[i+1]);

        //momentum adjustments
        for(int i=1;i<layerCount;i++)
            layers[i].momentum(alpha);
        
        //weight adjustments
        for(int i=1;i<layerCount;i++)
            layers[i].correction(alpha, beta, layers[i-1]);
    }

    private void bpgt(double[] in, double[] tgt)
    {
        ffwd(in);
        bkwd(tgt);
        //printweights();
    }

    public void printweights()
    {
        String s = "";
        Layer layer = layers[1];
        int neuroCount = layer.getNeuroneCount();
        s = s + "[";
        for(int i=0;i<neuroCount;i++)
        {
            Neurone n = layer.getNeurone(i);
            double[] weights = n.getPrevDwts();
            s = s + "(";
            for(double value:weights)
            {
                s = s + value + ",";
            }
            s = s  + ") ";
        }
        s = s + "],";
        System.out.println(s);        
    }

    public void train()
    {
        List<Double> errorData = new ArrayList<Double>();
        int dataSize = input.length;
        for(int i=0;i<iterCount;i++)
        {
            bpgt(input[i % dataSize], output[i % dataSize]);
            double error = mse(output[i % dataSize]);
            errorData.add(error);
            if(error <= threshold)
                break;
        }
        for(double value:errorData)System.out.println(value);
    }

    public double[] test(double[] data)
    {
        double[] out = null;
        ffwd(data);
        int count = outputLayer.getNeuroneCount();
        out = new double[count];
        for(int i=0;i<count;i++)
            out[i] = outputLayer.getNeurone(i).getOut();
        return out;
    }
}
