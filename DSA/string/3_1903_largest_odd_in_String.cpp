#include <string>
using namespace std;
class Solution {
public:
    string largestOddNumber(string num) {
        int n = num.size();
        string ans="" ;
        for(int i=n-1;i>=0;i--){
            if(num[i]=='1' || num[i]=='3' ||num[i]=='5' ||num[i]=='7' ||num[i]=='9'){
                for(int j=0;j<=i;j++){
                    ans+=num[j];
                }
                return ans;
            }
        }
        return ans;
    }
};
// you can also do like this 
class Solution1 {
public:
    string largestOddNumber(string num) {
        // Traverse from rightmost to leftmost
        for (int i = num.size() - 1; i >= 0; i--) {
            int digit = num[i] - '0'; // convert char to int
            if (digit % 2 == 1) { // check odd
                return num.substr(0, i + 1); // take substring
            }
        }
        return ""; // no odd digit found
    }
};






