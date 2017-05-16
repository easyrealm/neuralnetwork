package neuronet;


public class Main
{

    public static void main(String[] args) 
    {
        double beta = 0.3;
        double alpha = 0.2;
        double threshold = 0.00001;
        int layerCount = 4;
        int[] layerConfig = {3,3,3,1};
        int iterCount = 100000;
        double[][] input = {{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
        double[][] target = {{0},{1},{1},{0},{1},{0},{0},{1}};

        Net net = new Net(beta, alpha, threshold, layerCount, layerConfig, iterCount, input, target);
        net.train();
        for(double[] in:input)
        {
            String s = "";
            double[] out = net.test(in);
            for(double o:out)
		s = s + o + " ";
            System.out.println(s+"\n");
        }
    }

}
