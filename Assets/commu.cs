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
    private int perframe = 9999999;
    private bool actioned = false;
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
        if ((count % perframe == 0) | actioned)
        {
            manager.Socket.Emit("chat", "User", "Sending Picture");
            manager.Socket.Emit("picture", capture());
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
        actioned = true;
    }

    byte capture()
    {
        return new byte();
    }
}
