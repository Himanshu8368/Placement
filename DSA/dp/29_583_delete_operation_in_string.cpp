#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class Solution {
public:
    int solve(int n,int m,string & word1,string & word2){
        vector<vector<int>> dp(n+1,vector<int>(m+1,0));
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                if(word1[i-1]==word2[j-1]){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[n][m];
    }
    int minDistance(string word1, string word2) {
        int n = word1.size();
        int m = word2.size();
        // vector<vector<int>> dp(n,vector<int>(m,-1));
        int ans =solve(n,m,word1,word2);
        // cout<<ans<<endl;

        return n+m-(2*ans);

    }
};