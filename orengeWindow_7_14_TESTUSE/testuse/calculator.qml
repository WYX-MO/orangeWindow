import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 400
    height: 600
    title: "高级计算器"
    color: "#f5f5f5"
    
    // 确保窗口居中
    Component.onCompleted: {
        mainWindow.x = (Screen.width - width) / 2
        mainWindow.y = (Screen.height - height) / 2
    }
    
    // 计算区域
    Rectangle {
        id: displayArea
        width: parent.width
        height: parent.height * 0.35
        color: "#2c3e50"
        radius: 0
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#34495e" }
            GradientStop { position: 1.0; color: "#2c3e50" }
        }
        
        // 显示文本
        Text {
            id: displayText
            text: calculator.get_display_text()
            color: "white"
            font.pixelSize = 50
            font.family = "Arial"
            horizontalAlignment: Text.AlignRight
            verticalAlignment: Text.AlignVCenter
            anchors {
                right: parent.right
                rightMargin: 20
                bottom: parent.bottom
                bottomMargin: 30
            }
            
            // 文本变化动画
            NumberAnimation on opacity {
                from: 0
                to: 1
                duration: 300
                easing.type: Easing.InOutQuad
                when: calculator.displayTextChanged
            }
        }
    }
    
    // 按钮区域
    Rectangle {
        id: buttonsArea
        width: parent.width
        height: parent.height - displayArea.height
        color: "#ecf0f1"
        radius: 0
        
        // 按钮网格
        Grid {
            id: buttonGrid
            columns: 4
            rows: 6
            spacing: 10
            anchors {
                fill: parent
                margins: 15
            }
            
            // 按钮样式
            style: GridStyle {
                backgroundColor: "#ecf0f1"
            }
            
            // 按钮工厂函数 
            function createButton(text, color, onClick) { 
                return Button { 
                    text: text 
                    width: buttonGrid.width / 4 - 15 
                    height: buttonGrid.height / 6 - 15 
                    background: Rectangle { 
                        radius: 25 
                        color: color 
                        border.color: "#bdc3c7" 
                        border.width: 0 
                        
                        // 按下效果 
                        Behavior on color { 
                            ColorAnimation { duration: 150 } 
                        } 
                    } 
                    
                    // 将 DropShadow 效果移到 Button 元素下，而不是 background 内 
                    DropShadow { 
                        anchors.fill: background 
                        color: "rgba(0, 0, 0, 0.2)" 
                        radius: 5 
                        x: 2 
                        y: 2 
                    } 
                    
                    font { 
                        family: "Arial" 
                        pixelSize: 24 
                    } 
                    color: "white" 
                    hoverEnabled: true 
                    
                    // 鼠标悬停效果 
                    onHoveredChanged: { 
                        if (hovered) { 
                            background.color = Qt.lighter(background.color, 1.1) 
                        } else { 
                            background.color = color 
                        } 
                    } 
                    
                    // 点击效果 
                    onPressAndHold: { 
                        background.color = Qt.darker(background.color, 1.2) 
                    } 
                    
                    onClicked: onClick 
                }
            }
            
            // 第一行按钮
            Component.onCompleted: {
                // 第一行
                buttonGrid.addItem(createButton("C", "#e74c3c", function() { calculator.buttonClicked("C") }))
                buttonGrid.addItem(createButton("⌫", "#e67e22", function() { calculator.buttonClicked("⌫") }))
                buttonGrid.addItem(createButton("√", "#e67e22", function() { calculator.buttonClicked("√") }))
                buttonGrid.addItem(createButton("/", "#e67e22", function() { calculator.buttonClicked("/") }))
                
                // 第二行
                buttonGrid.addItem(createButton("7", "#3498db", function() { calculator.buttonClicked("7") }))
                buttonGrid.addItem(createButton("8", "#3498db", function() { calculator.buttonClicked("8") }))
                buttonGrid.addItem(createButton("9", "#3498db", function() { calculator.buttonClicked("9") }))
                buttonGrid.addItem(createButton("*", "#e67e22", function() { calculator.buttonClicked("*") }))
                
                // 第三行
                buttonGrid.addItem(createButton("4", "#3498db", function() { calculator.buttonClicked("4") }))
                buttonGrid.addItem(createButton("5", "#3498db", function() { calculator.buttonClicked("5") }))
                buttonGrid.addItem(createButton("6", "#3498db", function() { calculator.buttonClicked("6") }))
                buttonGrid.addItem(createButton("-", "#e67e22", function() { calculator.buttonClicked("-") }))
                
                // 第四行
                buttonGrid.addItem(createButton("1", "#3498db", function() { calculator.buttonClicked("1") }))
                buttonGrid.addItem(createButton("2", "#3498db", function() { calculator.buttonClicked("2") }))
                buttonGrid.addItem(createButton("3", "#3498db", function() { calculator.buttonClicked("3") }))
                buttonGrid.addItem(createButton("+", "#e67e22", function() { calculator.buttonClicked("+") }))
                
                // 第五行
                buttonGrid.addItem(createButton("+/-", "#3498db", function() { calculator.buttonClicked("+/-") }))
                buttonGrid.addItem(createButton("0", "#3498db", function() { calculator.buttonClicked("0") }))
                buttonGrid.addItem(createButton(".", "#3498db", function() { calculator.buttonClicked(".") }))
                buttonGrid.addItem(createButton("=", "#2ecc71", function() { calculator.buttonClicked("=") }))
                
                // 第六行
                buttonGrid.addItem(createButton("1/x", "#e67e22", function() { calculator.buttonClicked("1/x") }))
                buttonGrid.addItem(createButton("%", "#e67e22", function() { calculator.buttonClicked("%") }))
                buttonGrid.addItem(createButton("", "#ecf0f1", function() {}))
                buttonGrid.addItem(createButton("", "#ecf0f1", function() {}))
            }
        }
    }
}