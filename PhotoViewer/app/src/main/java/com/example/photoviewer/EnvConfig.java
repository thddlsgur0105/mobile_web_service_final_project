package com.example.photoviewer;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class EnvConfig {
    private static final String TAG = "EnvConfig";
    private static Map<String, String> envMap = null;
    
    /**
     * .env 파일에서 환경 변수를 로드합니다.
     */
    public static void loadEnv(Context context) {
        if (envMap != null) {
            return; // 이미 로드됨
        }
        
        envMap = new HashMap<>();
        
        try {
            InputStream inputStream = context.getAssets().open(".env");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            
            String line;
            while ((line = reader.readLine()) != null) {
                // 주석 제거
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                
                // KEY=VALUE 형식 파싱
                int equalsIndex = line.indexOf('=');
                if (equalsIndex > 0) {
                    String key = line.substring(0, equalsIndex).trim();
                    String value = line.substring(equalsIndex + 1).trim();
                    
                    // 따옴표 제거 (있는 경우)
                    if ((value.startsWith("\"") && value.endsWith("\"")) ||
                        (value.startsWith("'") && value.endsWith("'"))) {
                        value = value.substring(1, value.length() - 1);
                    }
                    
                    envMap.put(key, value);
                }
            }
            
            reader.close();
            inputStream.close();
            
            Log.d(TAG, "Environment variables loaded: " + envMap.size() + " keys");
        } catch (IOException e) {
            Log.e(TAG, "Error loading .env file: " + e.getMessage());
            envMap = new HashMap<>(); // 빈 맵으로 초기화
        }
    }
    
    /**
     * 환경 변수 값을 가져옵니다.
     * @param key 환경 변수 키
     * @param defaultValue 기본값 (키가 없을 때 반환할 값)
     * @return 환경 변수 값 또는 기본값
     */
    public static String get(String key, String defaultValue) {
        if (envMap == null) {
            Log.w(TAG, "Environment variables not loaded. Call loadEnv() first.");
            return defaultValue;
        }
        
        String value = envMap.get(key);
        return value != null ? value : defaultValue;
    }
    
    /**
     * 환경 변수 값을 가져옵니다.
     * @param key 환경 변수 키
     * @return 환경 변수 값 또는 빈 문자열
     */
    public static String get(String key) {
        return get(key, "");
    }
}

