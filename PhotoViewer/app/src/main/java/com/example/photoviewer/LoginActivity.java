package com.example.photoviewer;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class LoginActivity extends AppCompatActivity {
    private static final String TAG = "LoginActivity";
    private static final String PREFS_NAME = "PhotoViewerPrefs";
    private static final String KEY_TOKEN = "auth_token";
    private static final String KEY_USERNAME = "username";
    
    private TextInputEditText usernameEditText;
    private TextInputEditText passwordEditText;
    private TextInputEditText emailEditText;
    private TextInputEditText passwordConfirmEditText;
    private MaterialButton loginButton;
    private MaterialButton registerButton;
    private View registerSection;
    private boolean isRegisterMode = false;
    
    private String siteUrl;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        
        // .env 파일에서 환경 변수 로드
        EnvConfig.loadEnv(this);
        // 로컬 서버 기본값: 에뮬레이터는 10.0.2.2, 실제 기기는 PC의 IP 주소 사용
        siteUrl = EnvConfig.get("SITE_URL", "http://127.0.0.1:8000");
        
        // 이미 로그인되어 있으면 MainActivity로 이동
        if (isLoggedIn()) {
            startMainActivity();
            return;
        }
        
        initViews();
        setupClickListeners();
    }
    
    private void initViews() {
        usernameEditText = findViewById(R.id.usernameEditText);
        passwordEditText = findViewById(R.id.passwordEditText);
        emailEditText = findViewById(R.id.emailEditText);
        passwordConfirmEditText = findViewById(R.id.passwordConfirmEditText);
        loginButton = findViewById(R.id.loginButton);
        registerButton = findViewById(R.id.registerButton);
        registerSection = findViewById(R.id.registerSection);
        
        // 초기 상태: 로그인 모드
        registerSection.setVisibility(View.GONE);
    }
    
    private void setupClickListeners() {
        // 로그인/회원가입 모드 전환
        MaterialButton toggleModeButton = findViewById(R.id.toggleModeButton);
        toggleModeButton.setOnClickListener(v -> toggleMode());
        
        // 로그인 버튼
        loginButton.setOnClickListener(v -> {
            String username = usernameEditText.getText() != null ? usernameEditText.getText().toString().trim() : "";
            String password = passwordEditText.getText() != null ? passwordEditText.getText().toString() : "";
            
            if (username.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "사용자명과 비밀번호를 입력해주세요.", Toast.LENGTH_SHORT).show();
                return;
            }
            
            new LoginTask().execute(username, password);
        });
        
        // 회원가입 버튼
        registerButton.setOnClickListener(v -> {
            String username = usernameEditText.getText() != null ? usernameEditText.getText().toString().trim() : "";
            String email = emailEditText.getText() != null ? emailEditText.getText().toString().trim() : "";
            String password = passwordEditText.getText() != null ? passwordEditText.getText().toString() : "";
            String passwordConfirm = passwordConfirmEditText.getText() != null ? passwordConfirmEditText.getText().toString() : "";
            
            if (username.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "사용자명과 비밀번호를 입력해주세요.", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (!password.equals(passwordConfirm)) {
                Toast.makeText(this, "비밀번호가 일치하지 않습니다.", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (password.length() < 8) {
                Toast.makeText(this, "비밀번호는 최소 8자 이상이어야 합니다.", Toast.LENGTH_SHORT).show();
                return;
            }
            
            new RegisterTask().execute(username, email, password, passwordConfirm);
        });
    }
    
    private void toggleMode() {
        isRegisterMode = !isRegisterMode;
        
        if (isRegisterMode) {
            // 회원가입 모드
            registerSection.setVisibility(View.VISIBLE);
            loginButton.setVisibility(View.GONE);
            registerButton.setVisibility(View.VISIBLE);
            MaterialButton toggleButton = findViewById(R.id.toggleModeButton);
            toggleButton.setText("이미 계정이 있으신가요? 로그인");
        } else {
            // 로그인 모드
            registerSection.setVisibility(View.GONE);
            loginButton.setVisibility(View.VISIBLE);
            registerButton.setVisibility(View.GONE);
            MaterialButton toggleButton = findViewById(R.id.toggleModeButton);
            toggleButton.setText("계정이 없으신가요? 회원가입");
        }
    }
    
    private boolean isLoggedIn() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String token = prefs.getString(KEY_TOKEN, "");
        return !token.isEmpty();
    }
    
    private void saveAuthInfo(String username, String token) {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putString(KEY_TOKEN, token);
        editor.putString(KEY_USERNAME, username);
        editor.apply();
        Log.d(TAG, "Auth info saved: username=" + username);
    }
    
    private void startMainActivity() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }
    
    // 로그인 AsyncTask
    private class LoginTask extends AsyncTask<String, Void, LoginResult> {
        @Override
        protected void onPreExecute() {
            loginButton.setEnabled(false);
            loginButton.setText("로그인 중...");
        }
        
        @Override
        protected LoginResult doInBackground(String... params) {
            String username = params[0];
            String password = params[1];
            
            try {
                String loginUrl = siteUrl + "/api_root/api-token-auth/";
                Log.d(TAG, "🔗 Login URL: " + loginUrl);
                Log.d(TAG, "👤 Username: " + username);
                
                URL url = new URL(loginUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setDoOutput(true);
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);
                
                // JSON 요청 본문 생성
                JSONObject requestBody = new JSONObject();
                requestBody.put("username", username);
                requestBody.put("password", password);
                
                String jsonBody = requestBody.toString();
                Log.d(TAG, "📤 Request body: " + jsonBody);
                
                // 요청 전송
                DataOutputStream os = new DataOutputStream(conn.getOutputStream());
                os.writeBytes(jsonBody);
                os.flush();
                os.close();
                
                int responseCode = conn.getResponseCode();
                Log.d(TAG, "📥 Response code: " + responseCode);
                
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    // 응답 읽기
                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    reader.close();
                    
                    String responseStr = response.toString();
                    Log.d(TAG, "✅ Response: " + responseStr);
                    
                    // JSON 파싱
                    JSONObject jsonResponse = new JSONObject(responseStr);
                    String token = jsonResponse.getString("token");
                    
                    conn.disconnect();
                    Log.d(TAG, "✅ Login successful, token received");
                    return new LoginResult(true, "로그인 성공!", username, token);
                } else {
                    // 에러 응답 읽기
                    String errorMessage = "HTTP " + responseCode;
                    try {
                        InputStream errorStream = conn.getErrorStream();
                        if (errorStream != null) {
                            BufferedReader errorReader = new BufferedReader(
                                    new InputStreamReader(errorStream, StandardCharsets.UTF_8));
                            StringBuilder errorResponse = new StringBuilder();
                            String line;
                            while ((line = errorReader.readLine()) != null) {
                                errorResponse.append(line);
                            }
                            errorReader.close();
                            errorMessage = errorResponse.toString();
                            Log.e(TAG, "❌ Error response: " + errorMessage);
                            
                            // JSON 에러 메시지 파싱 시도
                            try {
                                JSONObject errorJson = new JSONObject(errorMessage);
                                if (errorJson.has("non_field_errors")) {
                                    errorMessage = errorJson.getJSONArray("non_field_errors").getString(0);
                                } else if (errorJson.has("detail")) {
                                    errorMessage = errorJson.getString("detail");
                                }
                            } catch (Exception e) {
                                // JSON 파싱 실패 시 원본 메시지 사용
                            }
                        } else {
                            Log.e(TAG, "❌ Error stream is null");
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "❌ Error reading error stream: " + e.getMessage());
                    }
                    
                    conn.disconnect();
                    return new LoginResult(false, "로그인 실패 (" + responseCode + "): " + errorMessage, null, null);
                }
            } catch (java.net.UnknownHostException e) {
                Log.e(TAG, "❌ Network error - Unknown host: " + e.getMessage());
                return new LoginResult(false, "서버에 연결할 수 없습니다.\n서버 URL을 확인해주세요: " + siteUrl, null, null);
            } catch (java.net.SocketTimeoutException e) {
                Log.e(TAG, "❌ Network error - Timeout: " + e.getMessage());
                return new LoginResult(false, "서버 응답 시간이 초과되었습니다.\n네트워크 연결을 확인해주세요.", null, null);
            } catch (java.io.IOException e) {
                Log.e(TAG, "❌ Network error - IO: " + e.getMessage(), e);
                return new LoginResult(false, "네트워크 오류: " + e.getMessage(), null, null);
            } catch (Exception e) {
                Log.e(TAG, "❌ Login error: " + e.getMessage(), e);
                e.printStackTrace();
                return new LoginResult(false, "오류: " + e.getMessage() + "\n\n자세한 내용은 Logcat을 확인하세요.", null, null);
            }
        }
        
        @Override
        protected void onPostExecute(LoginResult result) {
            loginButton.setEnabled(true);
            loginButton.setText("로그인");
            
            if (result.success) {
                saveAuthInfo(result.username, result.token);
                Toast.makeText(LoginActivity.this, result.message, Toast.LENGTH_SHORT).show();
                startMainActivity();
            } else {
                Toast.makeText(LoginActivity.this, result.message, Toast.LENGTH_LONG).show();
            }
        }
    }
    
    // 회원가입 AsyncTask
    private class RegisterTask extends AsyncTask<String, Void, LoginResult> {
        @Override
        protected void onPreExecute() {
            registerButton.setEnabled(false);
            registerButton.setText("회원가입 중...");
        }
        
        @Override
        protected LoginResult doInBackground(String... params) {
            String username = params[0];
            String email = params[1];
            String password = params[2];
            String passwordConfirm = params[3];
            
            try {
                String registerUrl = siteUrl + "/api_root/register/";
                URL url = new URL(registerUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setDoOutput(true);
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);
                
                // JSON 요청 본문 생성
                JSONObject requestBody = new JSONObject();
                requestBody.put("username", username);
                if (!email.isEmpty()) {
                    requestBody.put("email", email);
                }
                requestBody.put("password", password);
                requestBody.put("password_confirm", passwordConfirm);
                
                String jsonBody = requestBody.toString();
                
                // 요청 전송
                DataOutputStream os = new DataOutputStream(conn.getOutputStream());
                os.writeBytes(jsonBody);
                os.flush();
                os.close();
                
                int responseCode = conn.getResponseCode();
                Log.d(TAG, "Register response code: " + responseCode);
                
                if (responseCode == HttpURLConnection.HTTP_CREATED || responseCode == HttpURLConnection.HTTP_OK) {
                    // 응답 읽기
                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    reader.close();
                    
                    // JSON 파싱
                    JSONObject jsonResponse = new JSONObject(response.toString());
                    String token = jsonResponse.getString("token");
                    
                    conn.disconnect();
                    return new LoginResult(true, "회원가입 성공!", username, token);
                } else {
                    // 에러 응답 읽기
                    BufferedReader errorReader = new BufferedReader(
                            new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8));
                    StringBuilder errorResponse = new StringBuilder();
                    String line;
                    while ((line = errorReader.readLine()) != null) {
                        errorResponse.append(line);
                    }
                    errorReader.close();
                    conn.disconnect();
                    
                    Log.e(TAG, "Register error: " + errorResponse.toString());
                    return new LoginResult(false, "회원가입 실패: " + errorResponse.toString(), null, null);
                }
            } catch (Exception e) {
                Log.e(TAG, "Register error: " + e.getMessage(), e);
                return new LoginResult(false, "오류: " + e.getMessage(), null, null);
            }
        }
        
        @Override
        protected void onPostExecute(LoginResult result) {
            registerButton.setEnabled(true);
            registerButton.setText("회원가입");
            
            if (result.success) {
                saveAuthInfo(result.username, result.token);
                Toast.makeText(LoginActivity.this, result.message, Toast.LENGTH_SHORT).show();
                startMainActivity();
            } else {
                Toast.makeText(LoginActivity.this, result.message, Toast.LENGTH_LONG).show();
            }
        }
    }
    
    // 로그인 결과 클래스
    private static class LoginResult {
        boolean success;
        String message;
        String username;
        String token;
        
        LoginResult(boolean success, String message, String username, String token) {
            this.success = success;
            this.message = message;
            this.username = username;
            this.token = token;
        }
    }
}

