from apps import create_app

app = create_app('config.Config')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
