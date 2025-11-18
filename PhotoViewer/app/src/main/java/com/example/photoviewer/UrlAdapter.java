package com.example.photoviewer;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;

import java.util.List;

public class UrlAdapter extends RecyclerView.Adapter<UrlAdapter.UrlViewHolder> {

    private final List<String> urlList;
    private OnUrlClickListener onUrlClickListener;
    private OnUrlDeleteListener onUrlDeleteListener;

    public interface OnUrlClickListener {
        void onUrlClick(String url);
    }

    public interface OnUrlDeleteListener {
        void onUrlDelete(String url, int position);
    }

    public void setOnUrlClickListener(OnUrlClickListener listener) {
        this.onUrlClickListener = listener;
    }

    public void setOnUrlDeleteListener(OnUrlDeleteListener listener) {
        this.onUrlDeleteListener = listener;
    }

    public UrlAdapter(List<String> urlList) {
        this.urlList = urlList;
    }

    @NonNull
    @Override
    public UrlViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_url, parent, false);
        return new UrlViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull UrlViewHolder holder, int position) {
        if (position >= urlList.size()) {
            return;
        }

        String url = urlList.get(position);
        holder.urlText.setText(url);

        // URL 클릭 - 브라우저로 이동
        holder.urlText.setOnClickListener(v -> {
            if (onUrlClickListener != null) {
                onUrlClickListener.onUrlClick(url);
            }
        });

        // 삭제 버튼 클릭
        holder.deleteButton.setOnClickListener(v -> {
            if (onUrlDeleteListener != null) {
                onUrlDeleteListener.onUrlDelete(url, position);
            }
        });
    }

    @Override
    public int getItemCount() {
        return urlList.size();
    }

    public static class UrlViewHolder extends RecyclerView.ViewHolder {
        TextView urlText;
        MaterialButton deleteButton;
        MaterialCardView cardView;

        public UrlViewHolder(@NonNull View itemView) {
            super(itemView);
            urlText = itemView.findViewById(R.id.urlText);
            deleteButton = itemView.findViewById(R.id.urlDeleteButton);
            cardView = (MaterialCardView) itemView;
        }
    }
}

