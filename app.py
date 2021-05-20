from dash_app import create_server

server = create_server()

if __name__ == '__main__':
    server.run_server(debug=True)