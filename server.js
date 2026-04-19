const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const os = require('os');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  maxHttpBufferSize: 1e7, 
  cors: {
    origin: "*"
  }
});

const PORT = process.env.PORT || 3000;


app.use(express.static(path.join(__dirname, 'public')));


app.get('/server-info', (req, res) => {
  const host = req.get('host');
  let mobileURL = "";


  if (host.includes('localhost') || host.includes('127.0.0.1')) {
    const interfaces = os.networkInterfaces();
    let localIP = 'localhost';
    
    for (let devName in interfaces) {
      interfaces[devName].forEach((details) => {
        if (details.family === 'IPv4' && !details.internal) {
          localIP = details.address;
        }
      });
    }
   
    mobileURL = `http://${localIP}:${PORT}/mobile.html`;
  } else {
    const protocol = req.protocol === 'http' && !host.includes('onrender.com') ? 'http' : 'https';
    mobileURL = `${protocol}://${host}/mobile.html`;
  }

  res.json({
    mobileURL: mobileURL
  });
});


io.on('connection', (socket) => {
  console.log('A user connected:', socket.id);


  socket.on('cameraFrame', (data) => {
    socket.broadcast.emit('cameraFrame', data);
  });

  
  socket.on('visionState', (data) => {
    socket.broadcast.emit('visionState', data);
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});


server.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`If running locally, check your QR code on the desktop page.`);
});