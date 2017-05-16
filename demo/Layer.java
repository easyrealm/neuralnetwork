package neuronet;

public class Layer
{
    private int index;
    private int neuroneCount;
    private Neurone[] neurones;
    private Net owner;

    public Layer(int index, int neuroneCount, Net owner)
    {
        this.index = index;
        this.neuroneCount = neuroneCount;
        this.owner = owner;
        init();
    }
    
    public int getNeuroneCount()
    {
        return neuroneCount;
    }

    private void init()
    {
        int wsize = 0;
        Layer prevLayer = owner.getLayer(index - 1);
        if(prevLayer != null )
            wsize = prevLayer.getNeuroneCount();
        neurones = new Neurone[neuroneCount];
        for(int i=0;i<neuroneCount;i++)
            neurones[i] = new Neurone(wsize, this);
    }
    private double sigmoid(double value)
    {
        return 1.0/(1.0 + Math.exp(-value));
    }

    public Neurone getNeurone(int index)
    {
        if((index >= 0)&&(index < neuroneCount))
            return neurones[index];
        else
            return null;
    }

    public void forward(Layer prevLayer)
    {
        int prevNeuroneCount = prevLayer.getNeuroneCount();
        for(Neurone neurone:neurones)
        {
            if(neurone.getWeightCount() == prevNeuroneCount)
            {
                double sum = 0;
                double[] weights = neurone.getWeights();
                for(int i=0;i<prevNeuroneCount;i++)
                {
                    sum += prevLayer.getNeurone(i).getOut() * weights[i];
                }
                sum += weights[prevNeuroneCount];
                neurone.setOut(sigmoid(sum));
            }
        }
    }

    public void backward(Layer nextLayer)
    {
        int nextLayerCount = nextLayer.getNeuroneCount();
        for(int i=0;i<neuroneCount;i++)
        {
            double sum = 0;
            for(int j=0;j<nextLayerCount;j++)
            {
                sum += nextLayer.getNeurone(j).getDelta() * nextLayer.getNeurone(j).getWeights()[i];
            }
            double out = getNeurone(i).getOut();
            double delta = out * (1 - out) * sum;
            getNeurone(i).setDelta(delta);
        }
    }

    public void momentum(double alpha)
    {
        for(Neurone neurone:neurones)
        {
            double[] weights = neurone.getWeights();
            double[] prevDwts = neurone.getPrevDwts();
            int weightCount = weights.length;
            for(int i=0;i<weightCount - 1;i++)
                weights[i] += alpha * prevDwts[i];
            //also the bias
            weights[weightCount - 1] += alpha * prevDwts[weightCount - 1];
        }
    }

    public void correction(double alpha, double beta, Layer prevLayer)
    {
        int prevLayerCount = prevLayer.getNeuroneCount();
        for(Neurone neurone:neurones)
        {
            if(prevLayerCount == neurone.getWeightCount())
            {
                double[] prevDwts = neurone.getPrevDwts();
                double[] weights = neurone.getWeights();
                for(int i=0;i<prevLayerCount;i++)
                {
                    prevDwts[i] = beta * neurone.getDelta() * prevLayer.getNeurone(i).getOut();
                    weights[i] += prevDwts[i];
                }
                prevDwts[prevLayerCount] = beta * neurone.getDelta();
                weights[prevLayerCount] += prevDwts[prevLayerCount];
            }
        }
    }
}
