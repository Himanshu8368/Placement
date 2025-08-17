#include <algorithm>
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    int longestSubsequence(int i,int j,int n,string & s,string & t,vector<vector<int>> & dp){
        if(i==n || j==n)return 0;
        if(s[i]==t[j]){
            return 1+ longestSubsequence(i+1,j+1,n,s,t,dp);
        }
        if(dp[i][j]!=-1) return dp[i][j];
        return dp[i][j]=max(longestSubsequence(i+1,j,n,s,t,dp),longestSubsequence(i,j+1,n,s,t,dp));
    }
    int longestPalindromeSubseq(string s) {
        string t =s;
        reverse(s.begin(),s.end());
        int n = s.size();
        vector<vector<int>> dp(n,vector<int>(n,-1));
        return longestSubsequence(0,0,n,s,t,dp);
    }
};