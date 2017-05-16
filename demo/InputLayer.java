package neuronet;


public class InputLayer extends Layer
{
    public InputLayer(int index, int neuroneCount, Net owner)
    {
        super(index, neuroneCount, owner);
    }
    public void forward(double[] in)
    {
        int neuroneCount = getNeuroneCount();
        if(in.length == neuroneCount)
        {
            for(int i=0;i<neuroneCount; i++)
                getNeurone(i).setOut(in[i]);
        }
    }
}
