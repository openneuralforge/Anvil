package blueprint

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"runtime"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
)

type ServerInfo struct {
	OS           string `json:"os"`
	RAM          string `json:"ram"`
	CPU          string `json:"cpu"`
	ComputerType string `json:"computerType"`
}

type Client struct {
	Conn     net.Conn
	Addr     string
	LastSeen time.Time
}

type Hub struct {
	addClientChan    chan *Client
	removeClientChan chan string
	clients          map[string]*Client
}

type InitializationRequest struct {
	Path   string `json:"path"`
	Amount string `json:"amount"`
}

// Declare hub as a global variable
var hub *Hub

func NewHub() *Hub {
	return &Hub{
		addClientChan:    make(chan *Client),
		removeClientChan: make(chan string),
		clients:          make(map[string]*Client),
	}
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.addClientChan:
			h.clients[client.Addr] = client
			fmt.Printf("Added client: %s\n", client.Addr)
		case addr := <-h.removeClientChan:
			delete(h.clients, addr)
			fmt.Printf("Removed client: %s\n", addr)
		}
	}
}

func (h *Hub) Broadcast(msg []byte) {
	for addr, client := range h.clients {
		_, err := client.Conn.Write(msg)
		if err != nil {
			fmt.Printf("Error broadcasting to client %s: %v\n", addr, err)
			h.RemoveClient(addr)
		} else {
			fmt.Println("msg sent to: ", client.Addr)
		}
	}
}

func (h *Hub) AddClient(client *Client) {
	h.addClientChan <- client
}

func (h *Hub) RemoveClient(addr string) {
	h.removeClientChan <- addr
}

func handleConnection(client *Client, password string) {
	defer client.Conn.Close()

	// Authentication
	buf := make([]byte, 1024)
	n, err := client.Conn.Read(buf)
	if err != nil {
		hub.RemoveClient(client.Addr)
		return
	}

	if string(buf[:n]) != password {
		fmt.Printf("Client %s provided wrong password\n", client.Addr)
		hub.RemoveClient(client.Addr)
		return
	}

	client.Conn.Write([]byte("Authenticated"))
	hub.AddClient(client) // Add the client to the hub.clients map

	addr := client.Conn.RemoteAddr().String()

	for {
		n, err := client.Conn.Read(buf)
		if err != nil {
			hub.RemoveClient(addr)
			return
		}

		client.LastSeen = time.Now()
		message := string(buf[:n])
		var response []byte
		fmt.Printf("Received data from %s: %s\n", addr, message)

		var js map[string]interface{}
		errCheckJson := json.Unmarshal([]byte(message), &js)
		fmt.Println(js)

		if errCheckJson != nil {
			switch message {
			case "ping":

				// Get OS info
				osInfo := runtime.GOOS

				// Get RAM info
				vmStat, err := mem.VirtualMemory()
				if err != nil {
					log.Fatalf("Error getting memory info: %v", err)
				}
				ramInfo := fmt.Sprintf("%.2fGB", float64(vmStat.Total)/(1024*1024*1024))

				// Get CPU info
				cpuInfo, err := cpu.Info()
				if err != nil {
					log.Fatalf("Error getting CPU info: %v", err)
				}
				cpuModel := ""
				if len(cpuInfo) > 0 {
					cpuModel = fmt.Sprintf("%d cores %s", cpuInfo[0].Cores, cpuInfo[0].ModelName)
				}

				/*serverInfo := ServerInfo{
					OS:           osInfo,
					RAM:          ramInfo,
					CPU:          cpuModel,
					ComputerType: "ai",
				}*/
				serverInfo := make(map[string]interface{})
				serverInfo["OS"] = osInfo
				serverInfo["RAM"] = ramInfo
				serverInfo["CPU"] = cpuModel
				serverInfo["ComputerType"] = "ai"
				serverInfo["type"] = "serverInfo"

				response, err = json.Marshal(serverInfo)
				if err != nil {
					fmt.Printf("Error marshaling JSON: %v\n", err)
					return
				}
				client.Conn.Write(response)

			default:
				response = []byte(`{"error": "unknown command"}`)
			}
		} else {
			if value, ok := js["type"]; ok {
				fmt.Println("Key exists, value: ", value)
				switch js["type"] {
				case "initializationPopulation":
					//initializationPopulation(js)
					break
				case "startEval":
					//fmt.Println("--------------starting eval-----------------------")
					//fmt.Println(js)
					//startEval(js)

					/*serverInfo := make(map[string]interface{})
					serverInfo["ffs"] = "osInfo"

					response, err = json.Marshal(serverInfo)
					if err != nil {
						fmt.Printf("Error marshaling JSON: %v\n", err)
						return
					}

					data := map[string]interface{}{
						"type":   "evalStatusUpdate",
						"bugger": "hello",
					}

					BroadcastJsonToClients(data)*/
					// Get some basic computer specs
					/*os := runtime.GOOS
					arch := runtime.GOARCH
					numCPU := runtime.NumCPU()

					result["cmd"] = "sysinfo"
					result["cachePath"] = "ai"
					result["os"] = os
					result["arch"] = arch
					result["numCPU"] = strconv.Itoa(numCPU)

					jsonData, _ := json.Marshal(result)

					c.WriteMessage(websocket.TextMessage, jsonData)*/
					break
				}
			} else {
				fmt.Println("Key does not exist")
			}
		}

		/*var responseData InitializationRequest
		err := json.Unmarshal([]byte(msg.Data), &responseData)
		if err != nil {
			log.Println("json unmarshal data:", err)
			break
		}*/

	}
}

func startTcpServer(password string) {
	hub = NewHub()
	go hub.Run()

	tlsConfig, err := generateTLSConfig()
	if err != nil {
		fmt.Printf("Error generating TLS config: %v\n", err)
		return
	}

	listener, err := tls.Listen("tcp", ":12346", tlsConfig)
	if err != nil {
		fmt.Printf("Error starting server: %v\n", err)
		return
	}
	defer listener.Close()

	fmt.Println("Server is listening on port 12346...")

	for {
		conn, err := listener.Accept()
		if err != nil {
			fmt.Printf("Error accepting connection: %v\n", err)
			continue
		}

		client := &Client{
			Conn: conn,
			Addr: conn.RemoteAddr().String(),
		}
		go handleConnection(client, password)
	}
}

func BroadcastJsonToClients(msg map[string]interface{}) error {
	// Marshal the message into JSON
	jsonData, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("error marshaling message: %v", err)
	}

	// Check if hub or hub.clients is nil
	if hub == nil {
		return fmt.Errorf("hub is nil")
	}
	if hub.clients == nil {
		return fmt.Errorf("hub.clients is nil")
	}

	// Loop through all connected clients and send the message
	for addr, client := range hub.clients {
		_, err := client.Conn.Write(jsonData)
		if err != nil {
			hub.RemoveClient(addr)
			return fmt.Errorf("error broadcasting to client %s: %v", addr, err)
		}
	}

	return nil
}
