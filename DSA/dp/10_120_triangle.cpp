#include <vector>
#include <algorithm>    
#include <climits>
using namespace std;

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        vector<vector<int>> dp(n,vector<int>(n,0));
        for(int i=0;i<n;i++){
            for(int j=0;j<=i;j++){
                if(i==0){
                    dp[0][0]=triangle[0][0];
                }
                else{
                    int left = INT_MAX;
                    int right = INT_MAX;

                    if(j>0){
                        left =dp[i-1][j-1];
                    }
                    if(j<i){
                        right=dp[i-1][j];
                    }
                    dp[i][j]= triangle[i][j]+min(left,right);
                }
            }
        }int mini =INT_MAX;
        for(int i=0;i<n;i++){
            mini = min(mini,dp[n-1][i]);
        }
        return mini;
    }
};