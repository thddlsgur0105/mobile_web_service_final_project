package com.example.photoviewer;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MenuItem;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;
import androidx.core.content.FileProvider;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    TextView textView;
    ProgressBar progressBar;
    SwipeRefreshLayout swipeRefreshLayout;
    RecyclerView recyclerView;
    ImageAdapter adapter;
    
    // LLM 챗봇 UI
    TextInputEditText llmInputText;
    MaterialButton llmSendButton;
    TextView llmResponseText;
    
    String siteUrl;
    String token = "";
    CloadImage taskDownload;
    
    // LLM API 설정 (OpenAI)
    String openaiApiKey;
    String openaiApiUrl = "https://api.openai.com/v1/chat/completions";
    LLMApiTask llmApiTask;
    
    private List<Bitmap> currentBitmaps = new ArrayList<>();
    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int PERMISSION_REQUEST_CODE = 100;
    private Uri selectedImageUri;
    
    // URL 메모장 관련
    private RecyclerView urlRecyclerView;
    private UrlAdapter urlAdapter;
    private List<String> urlList = new ArrayList<>();
    private LinearLayout urlBookmarkContent;
    private MaterialButton urlBookmarkToggle;
    private MaterialButton urlAddButton;
    private View urlBookmarkHeader;
    private boolean isUrlBookmarkExpanded = false;
    private static final String PREFS_NAME = "PhotoViewerPrefs";
    private static final String KEY_URL_LIST = "url_list";
    
    // 상단 영역 접기/펼치기 관련
    private LinearLayout topSectionContainer;
    private MaterialButton toggleTopSectionButton;
    private boolean isTopSectionExpanded = true;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // 🔹 .env 파일에서 환경 변수 로드
        EnvConfig.loadEnv(this);
        siteUrl = EnvConfig.get("SITE_URL", "https://thddlsgur01050331.pythonanywhere.com");
        openaiApiKey = EnvConfig.get("OPENAI_API_KEY", "");
        
        textView = findViewById(R.id.textView);
        progressBar = findViewById(R.id.progressBar);
        swipeRefreshLayout = findViewById(R.id.swipeRefreshLayout);
        recyclerView = findViewById(R.id.recyclerView);
        
        // 🔹 LLM 챗봇 UI 초기화
        llmInputText = findViewById(R.id.llmInputText);
        llmSendButton = findViewById(R.id.llmSendButton);
        llmResponseText = findViewById(R.id.llmResponseText);
        
        // 전송 버튼 클릭 이벤트
        llmSendButton.setOnClickListener(v -> processLLMQuestion());
        
        // 키보드에서 전송 버튼(Enter) 클릭 이벤트
        llmInputText.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND || 
                (event != null && event.getKeyCode() == KeyEvent.KEYCODE_ENTER && event.getAction() == KeyEvent.ACTION_DOWN)) {
                processLLMQuestion();
                return true;
            }
            return false;
        });
        
        // RecyclerView 초기 설정
        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        layoutManager.setOrientation(LinearLayoutManager.VERTICAL);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setHasFixedSize(false);
        recyclerView.setItemViewCacheSize(20);
        recyclerView.setNestedScrollingEnabled(false);  // SwipeRefreshLayout과 함께 사용할 때 필요
        
        // 🔹 Pull to Refresh 설정
        swipeRefreshLayout.setColorSchemeColors(
            Color.parseColor("#00BCD4"),  // Blue
            Color.parseColor("#4CAF50"),  // Green
            Color.parseColor("#FF9800"),  // Orange
            Color.parseColor("#F44336")   // Red
        );
        swipeRefreshLayout.setOnRefreshListener(new SwipeRefreshLayout.OnRefreshListener() {
            @Override
            public void onRefresh() {
                loadImages();
            }
        });

        // 🔹 권한 확인
        checkPermissions();
        
        // 🔹 상단 영역 접기/펼치기 초기화
        initTopSectionToggle();
        
        // 🔹 URL 메모장 초기화
        initUrlBookmark();
    }
    
    // 🔹 상단 영역 접기/펼치기 초기화
    private void initTopSectionToggle() {
        topSectionContainer = findViewById(R.id.topSectionContainer);
        toggleTopSectionButton = findViewById(R.id.toggleTopSectionButton);
        
        // 초기 상태 설정 (펼쳐진 상태)
        topSectionContainer.setAlpha(1.0f);
        topSectionContainer.setVisibility(View.VISIBLE);
        toggleTopSectionButton.setText("▼ 접기");
        isTopSectionExpanded = true;
        
        toggleTopSectionButton.setOnClickListener(v -> toggleTopSection());
    }
    
    // 🔹 상단 영역 접기/펼치기
    private void toggleTopSection() {
        isTopSectionExpanded = !isTopSectionExpanded;
        
        if (isTopSectionExpanded) {
            // 펼치기
            topSectionContainer.setVisibility(View.VISIBLE);
            toggleTopSectionButton.setText("▼ 접기");
            
            // 애니메이션
            topSectionContainer.animate()
                    .alpha(1.0f)
                    .setDuration(300)
                    .start();
        } else {
            // 접기
            topSectionContainer.animate()
                    .alpha(0.0f)
                    .setDuration(300)
                    .withEndAction(() -> topSectionContainer.setVisibility(View.GONE))
                    .start();
            toggleTopSectionButton.setText("▲ 펼치기");
        }
    }
    
    // 🔹 URL 메모장 초기화
    private void initUrlBookmark() {
        urlRecyclerView = findViewById(R.id.urlRecyclerView);
        urlBookmarkContent = findViewById(R.id.urlBookmarkContent);
        urlBookmarkToggle = findViewById(R.id.urlBookmarkToggle);
        urlAddButton = findViewById(R.id.urlAddButton);
        urlBookmarkHeader = findViewById(R.id.urlBookmarkHeader);
        
        // RecyclerView 설정
        LinearLayoutManager urlLayoutManager = new LinearLayoutManager(this);
        urlLayoutManager.setOrientation(LinearLayoutManager.VERTICAL);
        urlRecyclerView.setLayoutManager(urlLayoutManager);
        urlRecyclerView.setHasFixedSize(false);
        
        // URL 목록 로드
        loadUrlList();
        
        // 어댑터 설정
        urlAdapter = new UrlAdapter(urlList);
        urlAdapter.setOnUrlClickListener(url -> {
            // URL 클릭 시 브라우저로 이동
            try {
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    url = "https://" + url;
                }
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
                startActivity(intent);
            } catch (Exception e) {
                Toast.makeText(this, "URL을 열 수 없습니다: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                Log.e("PhotoViewer", "URL 열기 오류: " + e.getMessage());
            }
        });
        
        urlAdapter.setOnUrlDeleteListener((url, position) -> {
            // URL 삭제 확인
            new AlertDialog.Builder(this)
                    .setTitle("URL 삭제")
                    .setMessage("이 URL을 삭제하시겠습니까?\n" + url)
                    .setPositiveButton("삭제", (dialog, which) -> {
                        urlList.remove(position);
                        urlAdapter.notifyItemRemoved(position);
                        urlAdapter.notifyItemRangeChanged(position, urlList.size());
                        saveUrlList();
                        Toast.makeText(this, "URL이 삭제되었습니다.", Toast.LENGTH_SHORT).show();
                    })
                    .setNegativeButton("취소", null)
                    .show();
        });
        
        urlRecyclerView.setAdapter(urlAdapter);
        
        // 접기/펼치기 토글
        urlBookmarkHeader.setOnClickListener(v -> toggleUrlBookmark());
        urlBookmarkToggle.setOnClickListener(v -> toggleUrlBookmark());
        
        // URL 추가 버튼
        urlAddButton.setOnClickListener(v -> showAddUrlDialog());
    }
    
    // 🔹 URL 메모장 접기/펼치기
    private void toggleUrlBookmark() {
        isUrlBookmarkExpanded = !isUrlBookmarkExpanded;
        
        if (isUrlBookmarkExpanded) {
            urlBookmarkContent.setVisibility(View.VISIBLE);
            urlBookmarkToggle.setText("닫기");
        } else {
            urlBookmarkContent.setVisibility(View.GONE);
            urlBookmarkToggle.setText("열기");
        }
    }
    
    // 🔹 URL 추가 다이얼로그
    private void showAddUrlDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("URL 추가");
        
        final EditText input = new EditText(this);
        input.setHint("예: https://www.pinterest.com 또는 pinterest.com");
        input.setInputType(android.text.InputType.TYPE_TEXT_VARIATION_URI);
        builder.setView(input);
        
        builder.setPositiveButton("추가", (dialog, which) -> {
            String url = input.getText().toString().trim();
            if (!url.isEmpty()) {
                // URL 형식 검증 및 정규화
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    url = "https://" + url;
                }
                
                // 중복 체크
                if (urlList.contains(url)) {
                    Toast.makeText(this, "이미 등록된 URL입니다.", Toast.LENGTH_SHORT).show();
                    return;
                }
                
                urlList.add(url);
                urlAdapter.notifyItemInserted(urlList.size() - 1);
                saveUrlList();
                Toast.makeText(this, "URL이 추가되었습니다.", Toast.LENGTH_SHORT).show();
                
                // 펼쳐진 상태로 유지
                if (!isUrlBookmarkExpanded) {
                    toggleUrlBookmark();
                }
            } else {
                Toast.makeText(this, "URL을 입력해주세요.", Toast.LENGTH_SHORT).show();
            }
        });
        
        builder.setNegativeButton("취소", null);
        builder.show();
    }
    
    // 🔹 URL 목록 저장
    private void saveUrlList() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        
        JSONArray jsonArray = new JSONArray();
        for (String url : urlList) {
            jsonArray.put(url);
        }
        
        editor.putString(KEY_URL_LIST, jsonArray.toString());
        editor.apply();
    }
    
    // 🔹 URL 목록 로드
    private void loadUrlList() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String jsonString = prefs.getString(KEY_URL_LIST, "[]");
        
        try {
            JSONArray jsonArray = new JSONArray(jsonString);
            urlList.clear();
            for (int i = 0; i < jsonArray.length(); i++) {
                urlList.add(jsonArray.getString(i));
            }
        } catch (Exception e) {
            Log.e("PhotoViewer", "URL 목록 로드 오류: " + e.getMessage());
            urlList.clear();
        }
    }

    private void checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE
                        }, PERMISSION_REQUEST_CODE);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "권한이 허용되었습니다.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // 🔹 LLM 질문 처리 및 답변 생성
    private void processLLMQuestion() {
        String question = llmInputText.getText() != null ? llmInputText.getText().toString().trim() : "";
        
        if (question.isEmpty()) {
            llmResponseText.setText("질문을 입력해주세요.");
            return;
        }
        
        // 입력창 초기화
        llmInputText.setText("");
        
        // 질문 처리 중 표시
        llmResponseText.setText("처리 중...");
        
        // 기존 작업 취소
        if (llmApiTask != null && llmApiTask.getStatus() == AsyncTask.Status.RUNNING) {
            llmApiTask.cancel(true);
        }
        
        // LLM API 호출
        llmApiTask = new LLMApiTask();
        llmApiTask.execute(question);
    }
    
    // 🔹 LLM API 호출 AsyncTask
    private class LLMApiTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... questions) {
            String question = questions[0];
            int imageCount = currentBitmaps.size();
            
            // API 키가 없으면 기본 응답 반환
            if (openaiApiKey == null || openaiApiKey.isEmpty()) {
                return generateFallbackResponse(question, imageCount);
            }
            
            try {
                // OpenAI API 호출
                URL url = new URL(openaiApiUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setRequestProperty("Authorization", "Bearer " + openaiApiKey);
                conn.setDoOutput(true);
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(30000);
                
                // 시스템 프롬프트와 사용자 질문 구성
                String systemPrompt = "당신은 이미지 갤러리 앱의 어시스턴트입니다. " +
                        "현재 업로드된 이미지 개수는 " + imageCount + "개입니다. " +
                        "사용자의 질문에 친절하고 정확하게 답변해주세요. " +
                        "이미지 개수에 대한 질문이면 정확한 개수를 알려주세요.";
                
                // JSON 요청 본문 생성 (JSONObject 사용으로 안전하게 처리)
                JSONObject requestBody = new JSONObject();
                requestBody.put("model", "gpt-3.5-turbo");
                
                JSONArray messages = new JSONArray();
                
                JSONObject systemMessage = new JSONObject();
                systemMessage.put("role", "system");
                systemMessage.put("content", systemPrompt);
                messages.put(systemMessage);
                
                JSONObject userMessage = new JSONObject();
                userMessage.put("role", "user");
                userMessage.put("content", question);
                messages.put(userMessage);
                
                requestBody.put("messages", messages);
                requestBody.put("max_tokens", 500);
                requestBody.put("temperature", 0.7);
                
                String jsonBody = requestBody.toString();
                
                // 요청 전송
                DataOutputStream os = new DataOutputStream(conn.getOutputStream());
                os.writeBytes(jsonBody);
                os.flush();
                os.close();
                
                int responseCode = conn.getResponseCode();
                Log.d("PhotoViewer", "LLM API Response Code: " + responseCode);
                
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
                    
                    // JSON 파싱
                    JSONObject jsonResponse = new JSONObject(response.toString());
                    JSONArray choices = jsonResponse.getJSONArray("choices");
                    if (choices.length() > 0) {
                        JSONObject message = choices.getJSONObject(0).getJSONObject("message");
                        String content = message.getString("content");
                        conn.disconnect();
                        return content.trim();
                    }
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
                    
                    Log.e("PhotoViewer", "LLM API Error: " + errorResponse.toString());
                    return "API 호출 오류가 발생했습니다. 기본 응답을 표시합니다.\n\n" + 
                           generateFallbackResponse(question, imageCount);
                }
                
                conn.disconnect();
            } catch (Exception e) {
                Log.e("PhotoViewer", "LLM API 호출 오류: " + e.getMessage(), e);
                return "오류가 발생했습니다: " + e.getMessage() + "\n\n" + 
                       generateFallbackResponse(question, imageCount);
            }
            
            return generateFallbackResponse(question, imageCount);
        }
        
        @Override
        protected void onPostExecute(String response) {
            if (response != null && !response.isEmpty()) {
                llmResponseText.setText(response);
            } else {
                llmResponseText.setText("응답을 받을 수 없습니다.");
            }
        }
    }
    
    // 🔹 API 호출 실패 시 기본 응답 생성
    private String generateFallbackResponse(String question, int imageCount) {
        String lowerQuestion = question.toLowerCase();
        
        // 이미지 개수 관련 질문 패턴 인식
        if (lowerQuestion.contains("몇") || lowerQuestion.contains("개수") || 
            lowerQuestion.contains("개") || lowerQuestion.contains("수") ||
            lowerQuestion.contains("how many") || lowerQuestion.contains("count") ||
            lowerQuestion.contains("업로드") || lowerQuestion.contains("그림") ||
            lowerQuestion.contains("이미지") || lowerQuestion.contains("사진") ||
            lowerQuestion.contains("picture") || lowerQuestion.contains("image") ||
            lowerQuestion.contains("photo")) {
            
            if (imageCount == 0) {
                return "현재 업로드된 이미지가 없습니다. 동기화 버튼을 눌러 이미지를 불러오세요.";
            } else {
                return "현재 총 " + imageCount + "개의 이미지가 업로드되어 있습니다. ✨";
            }
        }
        
        // 인사 관련 질문
        if (lowerQuestion.contains("안녕") || lowerQuestion.contains("hello") || 
            lowerQuestion.contains("hi") || lowerQuestion.contains("반가")) {
            return "안녕하세요! 이미지 개수에 대해 물어보시면 도와드릴 수 있습니다. 😊";
        }
        
        // 도움말 관련 질문
        if (lowerQuestion.contains("도움") || lowerQuestion.contains("help") || 
            lowerQuestion.contains("뭐") || lowerQuestion.contains("무엇")) {
            return "이미지 개수에 대해 물어보실 수 있습니다. 예: \"몇 개의 이미지가 있어?\", \"업로드된 그림이 몇 개야?\"";
        }
        
        // 기본 응답
        return "현재 업로드된 이미지는 " + imageCount + "개입니다. 다른 질문이 있으시면 물어보세요!";
    }

    // 🔹 서버에서 이미지 목록 가져오기
    public void onClickDownload(View v) {
        loadImages();
    }

    private void loadImages() {
        if (taskDownload != null && taskDownload.getStatus() == AsyncTask.Status.RUNNING) {
            taskDownload.cancel(true);
        }
        taskDownload = new CloadImage();
        taskDownload.execute(siteUrl + "/api_root/Post/");
    }

    // 🔹 갤러리에서 이미지 선택
    public void onClickUpload(View v) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            selectedImageUri = data.getData();
            if (selectedImageUri != null) {
                Toast.makeText(this, "이미지 선택 완료", Toast.LENGTH_SHORT).show();
                new UploadImageTask().execute(selectedImageUri);
            }
        }
    }

    // 🔹 이미지 업로드
    private class UploadImageTask extends AsyncTask<Uri, Void, String> {
        @Override
        protected void onPreExecute() {
            progressBar.setVisibility(View.VISIBLE);
        }

        @Override
        protected String doInBackground(Uri... uris) {
            String uploadUrl = siteUrl + "/api_root/Post/";
            Uri imageUri = uris[0];
            String boundary = "*****";
            String LINE_FEED = "\r\n";
            try {
                URL url = new URL(uploadUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setUseCaches(false);
                conn.setDoOutput(true);
                conn.setDoInput(true);
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
                if (token != null && !token.isEmpty()) {
                    conn.setRequestProperty("Authorization", "Token " + token);
                }

                DataOutputStream request = new DataOutputStream(conn.getOutputStream());
                request.writeBytes("--" + boundary + LINE_FEED);
                request.writeBytes("Content-Disposition: form-data; name=\"image\"; filename=\"upload.jpg\"" + LINE_FEED);
                request.writeBytes("Content-Type: image/jpeg" + LINE_FEED);
                request.writeBytes(LINE_FEED);

                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    request.write(buffer, 0, bytesRead);
                }
                inputStream.close();

                request.writeBytes(LINE_FEED);
                request.writeBytes("--" + boundary + "--" + LINE_FEED);
                request.flush();
                request.close();

                int responseCode = conn.getResponseCode();
                Log.d("PhotoViewer", "Upload response code: " + responseCode);

                if (responseCode == HttpURLConnection.HTTP_CREATED || responseCode == HttpURLConnection.HTTP_OK) {
                    return "업로드 성공!";
                } else {
                    BufferedReader errorReader = new BufferedReader(new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8));
                    StringBuilder errorMsg = new StringBuilder();
                    String line;
                    while ((line = errorReader.readLine()) != null) {
                        errorMsg.append(line);
                    }
                    return "업로드 실패: " + errorMsg;
                }
            } catch (Exception e) {
                e.printStackTrace();
                return "오류: " + e.getMessage();
            }
        }

        @Override
        protected void onPostExecute(String result) {
            progressBar.setVisibility(View.GONE);
            Toast.makeText(getApplicationContext(), result, Toast.LENGTH_LONG).show();
            if (result.contains("성공")) {
                loadImages();
            }
        }
    }

    // 🔹 서버에서 이미지 다운로드
    private class CloadImage extends AsyncTask<String, Integer, List<Bitmap>> {
        @Override
        protected void onPreExecute() {
            progressBar.setVisibility(View.VISIBLE);
            swipeRefreshLayout.setRefreshing(true);
        }

        @Override
        protected List<Bitmap> doInBackground(String... urls) {
            List<Bitmap> bitmaps = new ArrayList<>();
            try {
                String apiUrl = urls[0];
                Log.d("PhotoViewer", "📡 Request to: " + apiUrl);
                URL url = new URL(apiUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                if (token != null && !token.isEmpty()) {
                    conn.setRequestProperty("Authorization", "Token " + token);
                }
                conn.setRequestProperty("Accept", "application/json");
                conn.connect();

                int responseCode = conn.getResponseCode();
                Log.d("PhotoViewer", "✅ API Response Code: " + responseCode);

                if (responseCode == 200) {
                    Log.d("PhotoViewer", "📥 Parsing JSON response...");
                    BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        sb.append(line);
                    }
                    reader.close();

                    String jsonString = sb.toString();
                    Log.d("PhotoViewer", "📄 JSON Response length: " + jsonString.length());
                    if (jsonString.length() > 0) {
                        Log.d("PhotoViewer", "📄 JSON Response (first 500 chars): " + jsonString.substring(0, Math.min(500, jsonString.length())));
                    }
                    
                    if (jsonString.trim().isEmpty()) {
                        Log.e("PhotoViewer", "❌ Empty JSON response");
                        return bitmaps;
                    }

                    JSONArray jsonArray = new JSONArray(jsonString);
                    Log.d("PhotoViewer", "📦 JSON Array size: " + jsonArray.length());
                    for (int i = 0; i < jsonArray.length(); i++) {
                        JSONObject post = jsonArray.getJSONObject(i);
                        if (post.has("image")) {
                            String imageUrl = post.getString("image");

                            if (imageUrl == null || imageUrl.equals("null") || imageUrl.isEmpty()) {
                                Log.d("PhotoViewer", "⚠️ Empty image URL, skipping...");
                                continue;
                            }

                            if (!imageUrl.startsWith("http")) {
                                imageUrl = siteUrl + imageUrl;
                            }

                            Log.d("PhotoViewer", "🖼 Image URL: " + imageUrl);

                            try {
                                URL imgUrl = new URL(imageUrl);
                                HttpURLConnection imgConn = (HttpURLConnection) imgUrl.openConnection();
                                imgConn.setConnectTimeout(15000);
                                imgConn.setReadTimeout(15000);
                                imgConn.setRequestMethod("GET");
                                imgConn.connect();
                                
                                int imgResponseCode = imgConn.getResponseCode();
                                Log.d("PhotoViewer", "📸 Image Response Code: " + imgResponseCode);
                                
                                if (imgResponseCode == 200) {
                                    InputStream is = imgConn.getInputStream();
                                    Bitmap bitmap = BitmapFactory.decodeStream(is);
                                    if (bitmap != null) {
                                        bitmaps.add(bitmap);
                                        Log.d("PhotoViewer", "✅ Image loaded successfully");
                                    } else {
                                        Log.e("PhotoViewer", "❌ Bitmap is null");
                                    }
                                    is.close();
                                } else {
                                    Log.e("PhotoViewer", "❌ Image load failed: " + imgResponseCode);
                                }
                                imgConn.disconnect();
                            } catch (Exception e) {
                                Log.e("PhotoViewer", "❌ Error loading image: " + e.getMessage(), e);
                            }
                        }
                    }
                    Log.d("PhotoViewer", "📊 Total images loaded: " + bitmaps.size());
                } else {
                    Log.e("PhotoViewer", "❌ API Error: Response code " + responseCode);
                    try {
                        BufferedReader errorReader = new BufferedReader(new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8));
                        StringBuilder errorMsg = new StringBuilder();
                        String line;
                        while ((line = errorReader.readLine()) != null) {
                            errorMsg.append(line);
                        }
                        Log.e("PhotoViewer", "❌ Error message: " + errorMsg.toString());
                    } catch (Exception e) {
                        Log.e("PhotoViewer", "❌ Failed to read error stream: " + e.getMessage());
                    }
                }
                conn.disconnect();
            } catch (Exception e) {
                Log.e("PhotoViewer", "❌ Fatal Error: " + e.getMessage(), e);
                e.printStackTrace();
            }
            return bitmaps;
        }

        @Override
        protected void onPostExecute(List<Bitmap> bitmaps) {
            progressBar.setVisibility(View.GONE);
            swipeRefreshLayout.setRefreshing(false);

            currentBitmaps = bitmaps;

            if (bitmaps.isEmpty()) {
                Toast.makeText(getApplicationContext(), "표시할 이미지가 없습니다.", Toast.LENGTH_SHORT).show();
                textView.setText("서버에 이미지가 없습니다.");
                adapter = new ImageAdapter(new ArrayList<>());
                recyclerView.setAdapter(adapter);
            } else {
                Toast.makeText(getApplicationContext(), bitmaps.size() + "개의 이미지를 불러왔습니다.", Toast.LENGTH_SHORT).show();

                adapter = new ImageAdapter(bitmaps);
                
                // 🔹 이미지 클릭/롱클릭 이벤트 설정
                adapter.setOnImageClickListener(new ImageAdapter.OnImageClickListener() {
                    @Override
                    public void onImageClick(Bitmap bitmap, int position) {
                        // 🔹 이미지 상세보기 (확대/줌)
                        // Bitmap을 파일로 저장하고 URI 전달 (크기 제한 회피)
                        try {
                            File cacheDir = getCacheDir();
                            // 이전 파일 삭제 (캐시 정리)
                            File[] oldFiles = cacheDir.listFiles((dir, name) -> name.startsWith("detail_image_"));
                            if (oldFiles != null) {
                                for (File oldFile : oldFiles) {
                                    oldFile.delete();
                                }
                            }
                            
                            File imageFile = new File(cacheDir, "detail_image_" + position + "_" + System.currentTimeMillis() + ".jpg");
                            FileOutputStream fos = new FileOutputStream(imageFile);
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
                            fos.flush();
                            fos.close();
                            
                            Log.d("PhotoViewer", "💾 Image saved to: " + imageFile.getAbsolutePath());
                            
                            Uri imageUri = FileProvider.getUriForFile(MainActivity.this,
                                    "com.example.photoviewer.fileprovider", imageFile);
                            
                            Log.d("PhotoViewer", "🔗 FileProvider URI: " + imageUri.toString());
                            
                            Intent intent = new Intent(MainActivity.this, ImageDetailActivity.class);
                            intent.putExtra("image_uri", imageUri.toString());
                            intent.putExtra("position", position);
                            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                            startActivity(intent);
                        } catch (Exception e) {
                            Log.e("PhotoViewer", "❌ Error saving bitmap for detail view: " + e.getMessage(), e);
                            e.printStackTrace();
                            Toast.makeText(MainActivity.this, "이미지 열기 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        }
                    }

                    @Override
                    public void onImageLongClick(Bitmap bitmap, int position, View view) {
                        // 🔹 공유/저장 메뉴 표시
                        showImageMenu(bitmap, view);
                    }
                });
                
                recyclerView.setAdapter(adapter);
                adapter.notifyDataSetChanged();
                recyclerView.post(() -> {
                    recyclerView.smoothScrollToPosition(0);
                });
                textView.setText("✨ 총 " + bitmaps.size() + "개의 이미지가 표시됩니다.");
                Log.d("PhotoViewer", "✅ Adapter set with " + bitmaps.size() + " images");
            }
        }
    }

    // 🔹 이미지 공유/저장 메뉴
    private void showImageMenu(Bitmap bitmap, View view) {
        PopupMenu popupMenu = new PopupMenu(this, view);
        popupMenu.getMenu().add("공유하기");
        popupMenu.getMenu().add("저장하기");
        
        popupMenu.setOnMenuItemClickListener(item -> {
            if (item.getTitle().equals("공유하기")) {
                shareImage(bitmap);
            } else if (item.getTitle().equals("저장하기")) {
                saveImage(bitmap);
            }
            return true;
        });
        
        popupMenu.show();
    }

    // 🔹 이미지 공유 기능
    private void shareImage(Bitmap bitmap) {
        try {
            File cachePath = new File(getCacheDir(), "images");
            cachePath.mkdirs();
            File file = new File(cachePath, "share_image_" + System.currentTimeMillis() + ".jpg");
            
            FileOutputStream fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();

            Uri imageUri = FileProvider.getUriForFile(this,
                    "com.example.photoviewer.fileprovider", file);

            Intent shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.setType("image/jpeg");
            shareIntent.putExtra(Intent.EXTRA_STREAM, imageUri);
            shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(shareIntent, "이미지 공유하기"));
        } catch (Exception e) {
            Toast.makeText(this, "공유 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            Log.e("PhotoViewer", "Share error: " + e.getMessage());
        }
    }

    // 🔹 이미지 저장 기능
    private void saveImage(Bitmap bitmap) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10 이상
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, "PhotoViewer_" + System.currentTimeMillis() + ".jpg");
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/PhotoViewer");

            Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri != null) {
                try {
                    java.io.OutputStream out = getContentResolver().openOutputStream(uri);
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                    out.close();
                    Toast.makeText(this, "이미지가 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show();
                } catch (Exception e) {
                    Toast.makeText(this, "저장 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    Log.e("PhotoViewer", "Save error: " + e.getMessage());
                }
            }
        } else {
            // Android 9 이하
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
                return;
            }

            try {
                File picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
                File photoViewerDir = new File(picturesDir, "PhotoViewer");
                if (!photoViewerDir.exists()) {
                    photoViewerDir.mkdirs();
                }

                File file = new File(photoViewerDir, "PhotoViewer_" + System.currentTimeMillis() + ".jpg");
                FileOutputStream fos = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
                fos.flush();
                fos.close();

                Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                mediaScanIntent.setData(Uri.fromFile(file));
                sendBroadcast(mediaScanIntent);

                Toast.makeText(this, "이미지가 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show();
            } catch (Exception e) {
                Toast.makeText(this, "저장 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                Log.e("PhotoViewer", "Save error: " + e.getMessage());
            }
        }
    }
}

