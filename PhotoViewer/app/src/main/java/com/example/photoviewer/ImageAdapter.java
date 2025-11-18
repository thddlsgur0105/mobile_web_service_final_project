package com.example.photoviewer;

import android.animation.ObjectAnimator;
import android.content.Intent;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AccelerateDecelerateInterpolator;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.card.MaterialCardView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class ImageAdapter extends RecyclerView.Adapter<ImageAdapter.ImageViewHolder> {

    private final List<Bitmap> imageList;
    private OnImageClickListener onImageClickListener;

    public interface OnImageClickListener {
        void onImageClick(Bitmap bitmap, int position);
        void onImageLongClick(Bitmap bitmap, int position, View view);
    }

    public void setOnImageClickListener(OnImageClickListener listener) {
        this.onImageClickListener = listener;
    }

    public ImageAdapter(List<Bitmap> imageList) {
        this.imageList = imageList;
    }

    @NonNull
    @Override
    public ImageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_image, parent, false);
        return new ImageViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ImageViewHolder holder, int position) {
        if (position >= imageList.size()) {
            return;
        }
        
        Bitmap bitmap = imageList.get(position);
        if (bitmap != null && !bitmap.isRecycled()) {
            holder.imageView.setImageBitmap(bitmap);
            
            // 페이드 인 애니메이션
            holder.cardView.setAlpha(0f);
            ObjectAnimator fadeIn = ObjectAnimator.ofFloat(holder.cardView, "alpha", 0f, 1f);
            fadeIn.setDuration(400);
            fadeIn.setStartDelay(position * 100);
            fadeIn.setInterpolator(new AccelerateDecelerateInterpolator());
            fadeIn.start();
            
            // 클릭 이벤트 - 이미지 상세보기
            holder.cardView.setOnClickListener(v -> {
                if (onImageClickListener != null) {
                    onImageClickListener.onImageClick(bitmap, position);
                }
            });

            // 롱클릭 이벤트 - 공유/저장 메뉴
            holder.cardView.setOnLongClickListener(v -> {
                if (onImageClickListener != null) {
                    onImageClickListener.onImageLongClick(bitmap, position, v);
                }
                return true;
            });
        }
    }

    @Override
    public int getItemCount() {
        return imageList.size();
    }

    public static class ImageViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;
        MaterialCardView cardView;

        public ImageViewHolder(@NonNull View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.imageViewItem);
            cardView = (MaterialCardView) itemView;
        }
    }
}


