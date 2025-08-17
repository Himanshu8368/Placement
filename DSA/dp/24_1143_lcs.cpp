#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    int solve(int i, int j, int n, int m, string &text1, string &text2,vector<vector<int>> & dp) {
        if (i == n || j == m) return 0;

        if (dp[i][j] != -1) return dp[i][j];

        if (text1[i] == text2[j])
            return dp[i][j]= 1 + solve(i+1, j+1, n, m, text1, text2,dp);
        else
            return dp[i][j]= max(solve(i+1, j, n, m, text1, text2,dp),solve(i, j+1, n, m, text1, text2,dp));
    }
    //tabulation
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

    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size();
        int m = text2.size();
        vector<vector<int>> dp(n, vector<int>(m, -1));
        return solve(0, 0, n, m, text1, text2,dp);

    }
};
