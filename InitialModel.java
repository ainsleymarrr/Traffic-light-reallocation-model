public class InitialModel {
    static int ts=5;
    static double s=2.5;
    static int maxg=70;
    static int ming=20;
    static int lost=5;
    static int cycle=95;
    static double weight=8;
    // ts = 5 seconds, so 19 timesteps per 95-second cycle

    static int[][] data1 = {
    // Cycle 1: pre-dismissal
    {1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7},

    // Cycle 2: bell rings (sharp rise)
    {6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15},

    // Cycle 3: peak
    {14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,22,22},

    // Cycle 4: sustained peak
    {16,16,17,17,18,18,18,19,19,19,18,18,18,17,17,16,16,15,15},

    // Cycle 5: taper begins
    {13,13,14,14,14,15,15,14,14,13,13,12,12,11,11,10,10,9,9},

    // Cycle 6: taper
    {8,8,9,9,9,10,9,9,8,8,7,7,6,6,6,5,5,4,4},

    // Cycle 7: post-dismissal
    {3,3,4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,1,1},

    // Cycle 8: normal
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
    };

    static int[][] data2 = {
    // Cycle 1
    {1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4},

    // Cycle 2
    {3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,7,7,8,8},

    // Cycle 3
    {7,7,8,8,9,9,9,10,10,10,11,11,11,11,12,12,12,12,12},

    // Cycle 4
    {8,8,8,9,9,9,9,9,9,9,8,8,8,8,7,7,7,6,6},

    // Cycle 5
    {6,6,6,7,7,7,6,6,6,5,5,5,5,4,4,4,3,3,3},

    // Cycle 6
    {4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,2,1,1,1},

    // Cycle 7
    {2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1},

    // Cycle 8
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
    };






    //cars=arrival per timestep
    static double arrivalrate(int cars){
        return cars/(double)ts;
    }

    //pressure = original queue length + weight * arrival rate
    static double pressure(int cars, double w,int q){
        return q+w*arrivalrate(cars);
    }

    static int avg(int[] c){
        int car=0;
        for(int i:c) car+=i;
        return car/c.length;
    }

    static int[] green(int q1,int q2,int c1,int c2){
        //calculate pressure for each lane
        double pmain=pressure(c1,weight,q1);
        double psub=pressure(c2,weight,q2);
        int g1;
        int g2;

       g1=(int)((pmain/(pmain+psub)*(cycle-lost)));
       g1 = (int) Math.round(g1 / (double) ts) * ts;
       g1=Math.min(Math.max(g1,ming),maxg);
       g2=(cycle-lost)-g1;
        
       if (g2 < ming) {
            g2 = ming;
            g1 = (cycle - lost) - g2;
        }
        if (g2 > maxg) {
            g2 = maxg;
            g1 = (cycle - lost) - g2;
        }

        int[] greens=new int[2];
        greens[0]=g1;
        greens[1]=g2;

        return greens;
    }
    static boolean logTimesteps=false;
    //1. add ariving cars
    //2. decide which road is green and remove cars if green
    //cur: current timestep within the cycle
    //c1&c2, data for incoming cars for each timestep
    static double[] simulation(double q1, double q2, int[] c1, int[] c2, int[] green, int t, int cur,double delay,double totalarrival,String label,int ind){
        int g1=green[0];
        int g2=green[1];
        boolean data=true;
        if (logTimesteps) {
            System.out.println("\n======================================");
            System.out.println(label + " – Cycle " + (ind + 1));
            System.out.println("======================================");
            System.out.println("Time\tQ_main\tQ_side\tPhase\tCumDelay");
        }
        String phase;
        while(t<cycle){
            q1+=c1[cur];
            q2+=c2[cur];

            boolean lane1=(t<g1);
            boolean lane2=(t>=g1+lost&&t<g1+lost+g2);

            if (lane1) {
                phase = "Main";
            } else if (lane2) {
                phase = "Side";
            } else {
                phase = "Lost";
            }

            totalarrival+=c1[cur]+c2[cur];
            delay+=(q1+q2)*ts;

            if (logTimesteps) {
                System.out.printf(
                    "%d\t%.1f\t%.1f\t%s\t%.1f%n",
                    t, q1, q2, phase, delay
                );
            }

            
            double saturation=s*ts;
            
            if(lane1){
                double dep=Math.min(q1,saturation);
                q1-=dep;
            }else if(lane2){
                double dep=Math.min(q2,saturation);
                q2-=dep;
            }

            if(cur>=c1.length||cur>=c2.length){
                data=false;
                break;
            }

            t+=ts;
            cur+=1;

           

        }
        if(!data) System.out.print("Not enough data!!");
        return new double[]{q1,q2,t,cur,delay,totalarrival};
    }



    
    public static void main(String[] args) {
        double q1=10;
        double q2=8;
        double base1=q1;
        double base2=q2;
        int[] originalgreen={(cycle - lost)/2,cycle-lost-(cycle-lost)/2};
        double delay1=0;
        double arrival1=0;
        double delay2=0;
        double arrival2=0;
        // double[] compare=new double[data1.length];
        for(int z=0;z<data1.length;z++){
            
            int c1=avg(data1[z]);
            int c2=avg(data2[z]);

            int[] green=green((int)q1,(int)q2,c1,c2);
            double[] result1=simulation(q1, q2, data1[z], data2[z], green, 0, 0, 0, 0,"Adaptive",z);
            double[] result2=simulation(base1, base2, data1[z], data2[z], originalgreen, 0, 0, 0, 0,"Baseline",z);
            delay1+=result1[4];
            delay2+=result2[4];
            arrival1+=result1[5];
            arrival2+=result2[5];
            // compare[z]=avgdelay1-avgdelay2;
            q1=result1[0];
            q2=result1[1];
            base1=result2[0];
            base2=result2[1];

        }
        if(delay1/arrival1-delay2/arrival2<0) System.out.println("Effective model");
        else System.out.println("Needs improvement");
        System.out.println("Adaptive Model: "+delay1/arrival1);
        System.out.println("Baseline Model: "+delay2/arrival2);
        System.out.println("Percentage change: "+((delay2/arrival2-delay1/arrival1)/delay2/arrival2*100.0));

        
    }
}