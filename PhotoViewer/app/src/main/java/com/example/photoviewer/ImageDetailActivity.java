package com.example.photoviewer;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.github.chrisbanes.photoview.PhotoView;
import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class ImageDetailActivity extends AppCompatActivity {

    private Bitmap currentBitmap;
    private static final int PERMISSION_REQUEST_CODE = 200;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detail);

        MaterialToolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }

        PhotoView photoView = findViewById(R.id.photoView);
        MaterialButton btnShare = findViewById(R.id.btnShare);

        // MainActivity에서 전달된 URI에서 이미지 로드
        String imageUriString = getIntent().getStringExtra("image_uri");
        if (imageUriString != null) {
            try {
                Uri imageUri = Uri.parse(imageUriString);
                Log.d("PhotoViewer", "📸 Loading image from URI: " + imageUriString);
                
                // FileProvider URI를 통해 ContentResolver로 이미지 로드
                try {
                    java.io.InputStream inputStream = getContentResolver().openInputStream(imageUri);
                    if (inputStream != null) {
                        currentBitmap = BitmapFactory.decodeStream(inputStream);
                        inputStream.close();
                        
                        if (currentBitmap != null && !currentBitmap.isRecycled()) {
                            photoView.setImageBitmap(currentBitmap);
                            Log.d("PhotoViewer", "✅ Image loaded successfully from URI");
                        } else {
                            Log.e("PhotoViewer", "❌ Failed to decode bitmap or bitmap is recycled");
                            Toast.makeText(this, "이미지를 불러올 수 없습니다.", Toast.LENGTH_SHORT).show();
                        }
                    } else {
                        Log.e("PhotoViewer", "❌ InputStream is null");
                        Toast.makeText(this, "이미지를 불러올 수 없습니다.", Toast.LENGTH_SHORT).show();
                    }
                } catch (Exception e) {
                    Log.e("PhotoViewer", "❌ Error loading image from URI: " + e.getMessage(), e);
                    Toast.makeText(this, "이미지를 불러올 수 없습니다: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Log.e("PhotoViewer", "❌ Error parsing URI: " + e.getMessage(), e);
                Toast.makeText(this, "이미지를 불러올 수 없습니다: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        } else {
            // 기존 방식 지원 (fallback)
            currentBitmap = getIntent().getParcelableExtra("bitmap");
            if (currentBitmap != null && !currentBitmap.isRecycled()) {
                photoView.setImageBitmap(currentBitmap);
                Log.d("PhotoViewer", "✅ Image loaded from Parcelable (fallback)");
            }
        }

        // 공유 버튼 클릭
        btnShare.setOnClickListener(v -> {
            if (currentBitmap != null) {
                showShareSaveMenu();
            } else {
                Toast.makeText(this, "이미지를 불러올 수 없습니다.", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // 공유/저장 메뉴 표시
    private void showShareSaveMenu() {
        android.widget.PopupMenu popupMenu = new android.widget.PopupMenu(this, findViewById(R.id.btnShare));
        popupMenu.getMenu().add("공유하기");
        popupMenu.getMenu().add("저장하기");
        
        popupMenu.setOnMenuItemClickListener(item -> {
            if (item.getTitle().equals("공유하기")) {
                shareImage(currentBitmap);
            } else if (item.getTitle().equals("저장하기")) {
                saveImage(currentBitmap);
            }
            return true;
        });
        
        popupMenu.show();
    }

    // 이미지 공유 기능
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

    // 이미지 저장 기능
    private void saveImage(Bitmap bitmap) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
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

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
}


