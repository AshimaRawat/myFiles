
// Reading input till the End of file

package FIRST;

import java.util.*;

public class FIRST {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int count= 1;
        while(sc.hasNext())
        {
            System.out.println(count+""+sc.nextLine());
            count = count+1;
        }
    sc.close();
    }
    
}