package neuronet;

public class OutputLayer extends Layer
{
    public OutputLayer(int index, int neuroneCount, Net owner)
    {
        super(index, neuroneCount, owner);
    }
    public void backward(double[] tgt)
    {
        int neuroneCount = getNeuroneCount();
        for(int i=0;i<neuroneCount;i++)
        {
            double out = getNeurone(i).getOut();
            double delta = out * (1 - out)*(tgt[i] - out);
            getNeurone(i).setDelta(delta);
        }
    }
    
}
