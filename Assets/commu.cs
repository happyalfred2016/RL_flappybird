using System;
using System.Runtime.InteropServices;
using UnityEngine;
using BestHTTP.SocketIO;




// using BestHTTP;
// using UnityEngine.Purchasing;

public class commu : MonoBehaviour
{
    private SocketManager manager = new SocketManager(new Uri("http://127.0.0.1:5000/socket.io/"));
    private int count = 0;
    private int perframe = 60;
    private bool actioned = false;

    [DllImport("user32.dll", EntryPoint = "keybd_event")]
    public static extern void Keybd_event(
        byte bvk,//虚拟键值 ctrl键对应的是17
        byte bScan,//0
        int dwFlags,//0为按下，1按住，2释放
        int dwExtraInfo//0
    );
    
    void Start ()
    {
        manager.Open();
        // manager.Socket.Emit("chat", "userName", "message");
        manager.Socket.On("test", Ontest);
        manager.Socket.On("action", Onaction);
       
    }
    
    // Update is called once per frame
    void Update()
    {
        // TODO: wait until action reach.(In order to action with fixed time interval)
        if (actioned)
        {
            var bytes = capture();
            var status = getstatus();
            manager.Socket.Emit("chat", "User", "Sending Picture");
            manager.Socket.Emit("obs", bytes, status);
            actioned = false;
            count = 0;
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
        
        // In put.GetButtonDown("Fire1");
        if ((double)args[0]>0)
        {
            Keybd_event(17, 0, 0, 0);
            Keybd_event(17, 0, 2, 0);
        }
        
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