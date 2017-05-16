package neuronet;


public class Neurone
{
    private double out;
    private double delta;
    private int weightCount;
    private double[] weights = null;
    private double[] prevDwts = null;
    private Layer owner = null;

    public Neurone(int weightCount, Layer owner)
    {
        this.owner = owner;
        this.weightCount = weightCount;
        weights = new double[weightCount + 1]; //+1 bias
        prevDwts = new double[weightCount + 1]; //+1 bias
        //initialize the weights
        for(int i=0;i<weightCount + 1;i++)
        {
            weights[i] = (double)Net.random.nextInt(Net.RAND_MAX)/(double)(Net.RAND_MAX/2) - 1; //+1
            prevDwts[i] = 0.0;
        }
        //adding the bias
        weights[weightCount] = -1.0;
    }

    public double getOut() {
        return out;
    }

    public Layer getOwner() {
        return owner;
    }

    public int getWeightCount()
    {
        return weightCount;
    }

    public double[] getWeights()
    {
        return weights;
    }

    public double[] getPrevDwts() {
        return prevDwts;
    }
    

    public void setOut(double value)
    {
        out = value;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }
    
}
