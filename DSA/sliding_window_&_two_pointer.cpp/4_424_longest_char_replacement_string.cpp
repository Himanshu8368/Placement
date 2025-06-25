#include <string>
#include <algorithm>
using namespace std;
class Solution {
public:
    int characterReplacement(string s, int k) {
        int ans =0;
        for(int character = 'A';character<= 'z';character++){
            int left =0;
            int different =0;
            for(int right =0;right<s.size();right++){
                if(s[right]!=character) different++;
                while(different>k){
                    if(s[left]!= character){
                        different--;
                    }
                    left++;
                }
                ans = max(ans,right-left+1);
            }
        }
        return ans;
    }
};