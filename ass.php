<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multiplication Table</title>
    <style>
        body { font-family: Arial, sans-serif; }
        table { border-collapse: collapse; }
        th, td { border: 3px solid #999; padding: 12px; text-align: center; }
        th { background-color: #f6f7eb; }
        .color1 { background-color: #f4f1bb; }
        .color2 { background-color: #9bc1bc; }
    </style>
</head>
<body>

<table>
    <tr><th colspan="11" style="background-color:#f6f7eb;">MULTIPLICATION TABLE</th></tr>
    <tr style="background-color:#f6f7eb;">
        <th>x</th>
        <?php for ($header = 1; $header <= 10; $header++): ?>
            <th><?php echo $header ?></th>
        <?php endfor ?>
    </tr>

    <?php
    for ($row = 1; $row <= 10; $row++):
        echo '<tr>';
            echo '<td style="background-color:#f6f7eb;">' . $row . '</td>';
            for ($col = 1; $col <= 10; $col++):
                // Determine the color of the cell
                if (($row * $col) % 2 == 0) {
                    echo '<td class="color1">';
                } else {
                    echo '<td class="color2">';
                }

                // Display the product
                echo $row * $col;

                // Close the cell
                echo '</td>';
            endfor;
        echo '</tr>';
    endfor;
   

// Footer with full name and section

?>

<tr><th text-align: left colspan="11" style="background-color:#f6f7eb;">by Klyde Alexander V. Penus / TN22</th></tr>

</table>
</body>
</html>