using System;
using UnityEditor;
using UnityEngine;

public static class ImportDrawer
{
    private static int _framesPadding = 8;

    public static void Draw(Import import)
    {
        var controlId = GUIUtility.GetControlID(FocusType.Passive);
        var state = (ImportDrawerState)GUIUtility.GetStateObject(typeof(ImportDrawerState), controlId);
        var optimizedSprites = import.OptimizedSprites;

        if (Event.current.control && Event.current.type == EventType.ScrollWheel)
        {
            state.SizeFactor = Mathf.Clamp(state.SizeFactor - Mathf.Clamp(Mathf.RoundToInt(Event.current.delta.y), -1, 1), 1, int.MaxValue);
            Event.current.Use();
        }

        state.ScrollPosition = EditorGUILayout.BeginScrollView(state.ScrollPosition);

        for (int i = 0; i < optimizedSprites.Length; i++)
        {
            var optimizedSprite = optimizedSprites[i];

            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.Space();

            EditorGUILayout.BeginVertical();
            EditorGUILayout.LabelField($"Original:");
            drawInFrame(optimizedSprite, state, _framesPadding, spriteRect => DrawSprite(spriteRect, optimizedSprite.OriginalSprite, state.SizeFactor));
            EditorGUILayout.EndVertical();

            EditorGUILayout.BeginVertical();
            EditorGUILayout.LabelField($"Optimized:");
            drawInFrame(optimizedSprite, state, _framesPadding, spriteRect => DrawOptimizedSprite(spriteRect, optimizedSprite, state.SizeFactor));
            EditorGUILayout.EndVertical();

            EditorGUILayout.BeginVertical();
            EditorGUILayout.LabelField($"Chunks count: {optimizedSprite.Chunks.Length}");
            GUIStyle s = new GUIStyle(EditorStyles.label);
            var valid = optimizedSprite.Equals(optimizedSprite.OriginalSprite);
            s.normal.textColor = valid ? Color.green : Color.red;
            EditorGUILayout.LabelField($"Validation {(valid ? "passed" : "failed")}", s);
            EditorGUILayout.EndVertical();

            EditorGUILayout.EndHorizontal();
        }

        EditorGUILayout.EndScrollView();
    }

    private static void drawInFrame(OptimizedSprite optimizedSprite, ImportDrawerState state, int rightPadding, Action<Rect> drawSprite)
    {
        var reservedRectHeight = optimizedSprite.Height * state.SizeFactor + _framesPadding * 2;
        var reservedRectWidth = optimizedSprite.Width * state.SizeFactor + _framesPadding + rightPadding;
        var reservedSpriteRect = GUILayoutUtility.GetRect(reservedRectWidth, reservedRectHeight);
        //reservedSpriteRect.x = reservedSpriteRect.x + (reservedSpriteRect.width - reservedRectWidth) / 2;
        reservedSpriteRect.width = reservedRectWidth;

        Handles.BeginGUI();
        Handles.DrawSolidRectangleWithOutline(reservedSpriteRect, Color.clear, Color.black);
        Handles.EndGUI();
        var spriteRect = new Rect(reservedSpriteRect.position + Vector2.one * _framesPadding, new Vector2(optimizedSprite.Width, optimizedSprite.Height));
        drawSprite.Invoke(spriteRect);
    }

    private static void DrawSprite(Rect rect, Sprite sprite, int sizeFactor)
    {
        var wholeSpriteRect = sprite.rect;
        rect.size *= sizeFactor;
        if (Event.current.type == EventType.Repaint)
        {
            var texture = sprite.texture;
            var spriteRectInTexture = default(Rect);
            spriteRectInTexture.xMin = wholeSpriteRect.xMin / texture.width;
            spriteRectInTexture.xMax = wholeSpriteRect.xMax / texture.width;
            spriteRectInTexture.yMin = wholeSpriteRect.yMin / texture.height;
            spriteRectInTexture.yMax = wholeSpriteRect.yMax / texture.height;
            GUI.DrawTextureWithTexCoords(rect, texture, spriteRectInTexture);
        }
    }

    private static void DrawOptimizedSprite(Rect rect, OptimizedSprite optimizedSprite, int sizeFactor)
    {
        //var optimizedSpriteWidth = optimizedSprite.Width * sizeFactor;
        var optimizedSpriteHeight = optimizedSprite.Height * sizeFactor;
        //var allocatedRect = GUILayoutUtility.GetRect(optimizedSpriteWidth * sizeFactor, optimizedSpriteHeight);
        if (Event.current.type == EventType.Repaint)
        {
            //var finalRect = new Rect(allocatedRect.x + (allocatedRect.width - optimizedSpriteWidth) / 2, allocatedRect.y + (allocatedRect.height - optimizedSpriteHeight) / 2, optimizedSpriteWidth, optimizedSpriteHeight);
            var chunks = optimizedSprite.Chunks;
            for (int i = 0; i < chunks.Length; i++)
            {
                var chunk = chunks[i];
                var sprite = chunk.Sprite;
                var chunkSize = sprite.rect.size * sizeFactor;
                var spriteRect = sprite.rect;
                var texture = sprite.texture;
                var spriteRectInTexture = default(Rect);
                spriteRectInTexture.xMin = spriteRect.xMin / texture.width;
                spriteRectInTexture.xMax = spriteRect.xMax / texture.width;
                spriteRectInTexture.yMin = spriteRect.yMin / texture.height;
                spriteRectInTexture.yMax = spriteRect.yMax / texture.height;
                var chunkRect = new Rect(rect.position + new Vector2(chunk.X * sizeFactor, optimizedSpriteHeight - chunk.Y * sizeFactor - chunkSize.y), chunkSize);
                GUI.DrawTextureWithTexCoords(chunkRect, texture, spriteRectInTexture);
            }
        }
    }
}
