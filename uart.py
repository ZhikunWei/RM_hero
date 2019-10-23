import serial
import struct
import threading
import time


class Uart:
	def __init__(self):
		self.enemy_color = None
		self.angle = None
		self.predict = False
		self.uart = serial.Serial("/dev/ttyTHS2", 115200, timeout=0.5)
		if self.uart.isOpen():
			print("Uart open sucessfully")
		else:
			print("Uart close failed")
		t = threading.Thread(target=self.read)
		t.setDaemon(True)
		t.start()

	def openUart(self):
		self.uart = serial.Serial("/dev/ttyTHS2", 115200, timeout=0.5)  # uart1
		#self.uart.open()

	def sendTarget(self, x, y, z):
		x = int(x / 180.0* (32768 - 1) )
		y = int(y / 180.0* (32768 - 1) )
		z = int(z / 1000.0 * (32768 - 1) )
		
		#data = struct.pack('bhhh',0x7F, x, y, z)
		s = 0x7F.to_bytes(1, byteorder='big')
		if x < 0:
		    x += 0xFFFF
		if y < 0:
		    y += 0xFFFF
		x = x.to_bytes(2, byteorder='big')
		y = y.to_bytes(2, byteorder='big')
		z = z.to_bytes(2, byteorder='big')
		e = 0x26.to_bytes(1, byteorder='big')
		#print(x, y, z, s+x+y+z)
		#print(len(data),data, struct.unpack('bhhh', s+x+y+z))
		state = self.uart.write(s+x+y+z+e)
		#print(state)
		if state <= 0:
			self.uart.close()
			self.openUart()

	def read(self):
		while True:
			data = self.uart.readline()
			if len(data) < 4:
				time.sleep(0.005)
				continue
			angle = struct.unpack('bB', data[1:3])
			angle = angle[0] * 256 + angle[1]
			self.angle = angle / 32767.0 * 180
			color = struct.unpack('b', data[0:1])
			self.enemy_color = "red" if color[0] == 1 else "blue"
			predict = struct.unpack('b', data[3:4])
			self.predict = True if predict[0] == 1 else False
			time.sleep(0.01)
			
	
	def int2byte(self, data):   # https://www.cnblogs.com/sexyboy/p/8509666.html
		output = struct.pack('h', data)
		print(output)
		return output
	
	def byte2int(self, data):
		output = struct.unpack('bc', data)
		print(output)
		return output
		
	def __del__(self):
		self.uart.close()

if __name__ == '__main__':
	uart = Uart() 
	#data = uart.read()
	#print(uart.name, uart.port)
	i = 1
	while True:
		#i += 1
		#i = int(i % 10)
		#uart.sendTarget(0, 0, 500)
		#time.sleep(0.05)
		time.sleep(0.01)
		if uart.angle is not None:
		    print(time.time())
		    print(uart.angle, uart.enemy_color, uart.predict)
		    uart.angle = None



