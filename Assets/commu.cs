using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using BestHTTP.SocketIO;
using System;

using BestHTTP;
using UnityEngine.Purchasing;

public class commu : MonoBehaviour
{
    private SocketManager manager = new SocketManager(new Uri("http://127.0.0.1:5000/socket.io/"));
    private int count = 0;
    private int perframe = 60;
    private bool actioned = false;
    private bool activted = false;
    void Start ()
    {
        manager.Open();
        manager.Socket.Emit("chat", "userName", "message");
        manager.Socket.On("chat", Onchat);
        manager.Socket.On("action", Onaction);
       
    }
    
    // Update is called once per frame
    void Update()
    {
        // 
        // if (activted)
        // {
        //     if ((count % perframe == 0) & activted)
        //     {
        //         while(true)
        //             if (actioned)
        //             {
        //                 var bytes = capture();
        //                 manager.Socket.Emit("chat", "User", "Sending Picture");
        //                 manager.Socket.Emit("obs", bytes);
        //                 actioned = false;
        //                 // count = 0;
        //                 break;
        //             }
        //     }
        // }
        // else if (actioned)
        //     activted = true;
        
        // TODO: wait until action reach.(In order to action with fixed time interval)
        if (actioned)
        {
            var bytes = capture();
            var status = getstatus();
            manager.Socket.Emit("chat", "User", "Sending Picture");
            manager.Socket.Emit("obs", bytes, status);
            Debug.Log(status);
            actioned = false;
            count = 0;
        }
        count += 1;
    }
    void Onchat(Socket socket, Packet packet, params object[] args)
    {   
        Debug.Log("On Chat");
        Debug.Log(args);
    }
    
    void Onaction(Socket socket, Packet packet, params object[] args)
    {   
        Debug.Log("On Action");
        Debug.Log(args);
        
        
        // if ((int)args[0]>0)
        // {
        //     
        // }
        
        actioned = true;
    }

    byte[] capture()
    {
        int resWidth = 256; 
        int resHeight = 144;
        return CaptureScreenshot2(new Rect(0, 0, resWidth, resHeight));
    }

    int[] getstatus()
    {
        var prin = GameObject.FindObjectOfType<principal>();
        var pon = GameObject.FindObjectOfType<pontos>();
        return new int[]{Convert.ToInt32(prin.GetFim()), pon.i};
    }
    
    byte[] CaptureScreenshot2(Rect rect)
    {
        var camera = Camera.main;
        RenderTexture rt = new RenderTexture((int)rect.width, (int)rect.height, 24);
        camera.targetTexture = rt;
        Texture2D screenShot = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.ARGB32, false);
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
