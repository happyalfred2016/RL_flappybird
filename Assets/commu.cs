﻿using System;
using UnityEngine;
using BestHTTP.SocketIO;

// using BestHTTP;
// using UnityEngine.Purchasing;

public class commu : MonoBehaviour
{
    private SocketManager manager = new SocketManager(new Uri("http://127.0.0.1:5000/socket.io/"));
    private int count = 0;
    private int perframe = 10;
    private bool actioned = false;
    private int actionDo = 0;
    private bool isready = false;

    principal prin;
    pontos pon;


    void Start()
    {
        manager.Open();
        // manager.Socket.Emit("chat", "userName", "message");
        manager.Socket.On("test", Ontest);
        manager.Socket.On("action", Onaction);
        prin = GameObject.FindObjectOfType<principal>();
        pon = GameObject.FindObjectOfType<pontos>();
        Application.targetFrameRate = 60;
    }

    // Update is called once per frame
    void Update()
    {
        // TODO: wait until action reach.(In order to action with fixed time interval)
        if (count % perframe == 0)
        {
            Time.timeScale = 0;
            if (actioned)
            {
                // In put.GetButtonDown("Fire1");
                // Time.timeScale = 0;
                if (actionDo > 0)
                    prin.man_tri();
                var bytes = capture();
                var status = getstatus();
                manager.Socket.Emit("chat", "User", "Sending Picture");
                manager.Socket.Emit("obs", bytes, status);
                actioned = false;
                actionDo = 0;
                count = 0;
                Time.timeScale = 1;
            }
        }

        count += 1;
    }

    void Ontest(Socket socket, Packet packet, params object[] args)
    {
        Debug.Log("On Test");
        Debug.Log(args[0]);
    }

    void Onaction(Socket socket, Packet packet, params object[] args)
    {
        Debug.Log("On Action");
        Debug.Log(args[0]);
        actionDo = Convert.ToInt32(args[0]);
        actioned = true;
    }

    byte[] capture()
    {
        int resWidth = 64;
        int resHeight = 36;
        return CaptureScreenshot2(new Rect(0, 0, resWidth, resHeight));
    }

    int[] getstatus()
    {
        return new int[] {Convert.ToInt32(prin.GetFim()), pon.i};
    }

    byte[] CaptureScreenshot2(Rect rect)
    {
        var camera = Camera.main;
        RenderTexture rt = new RenderTexture((int) rect.width, (int) rect.height, 24);
        camera.targetTexture = rt;
        Texture2D screenShot = new Texture2D((int) rect.width, (int) rect.height, TextureFormat.ARGB32, false);
        camera.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(rect, 0, 0);
        camera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        byte[] bytes = screenShot.EncodeToPNG();
        // string filename = Application.dataPath + "/Screenshot.png";  
        //
        // System.IO.File.WriteAllBytes(filename, bytes);
        // Debug.Log(string.Format("Took screenshot to: {0}", filename));
        // // Application.OpenURL(filename);
        return bytes;
    }
}