use actix_web::{web, App, HttpResponse, HttpServer};
use actix_web::Responder;
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use serde::Deserialize;
use actix_web_prom::PrometheusMetrics;
use actix_web_prom::PrometheusMetricsBuilder;
use prometheus::{opts, IntCounterVec};

#[derive(Deserialize)]
struct Inquiry {
    question: String,
    context: String,
}
async fn process_qa(correct_counter: web::Data<IntCounterVec>, payload: web::Json<Inquiry>) -> impl Responder {
    let result = web::block(move || {
        let model = QuestionAnsweringModel::new(Default::default()).expect("Failed to create model");
        let queries = [QaInput {
            question: payload.question.clone(),
            context: payload.context.clone(),
        }];
        model.predict(&queries, 1, 32)
    })
    .await;

    match result {
        Ok(answers) => {
            let response_text = if let Some(first_answer) = answers.first() {
                // Adjust this line to correctly print the fields of `first_answer`, ensure the fields exist
                // For example, using `format!("Answer: {:?}", first_answer)` if you want to print all fields
                format!("Answer: {:?}", first_answer)  
            } else {
                String::from("No valid answer found.")
            };
            correct_counter.with_label_values(&["/qa"]).inc();
            HttpResponse::Ok().content_type("text/plain").body(response_text)
        },
        Err(_) => HttpResponse::InternalServerError().content_type("text/plain").body("There was a problem processing the request.")
    }
}

fn setup_prometheus_metrics() -> (PrometheusMetrics, IntCounterVec) {
    let metrics = PrometheusMetricsBuilder::new("api")
        .endpoint("/metrics")
        .build()
        .expect("Failed to build Prometheus metrics");

    let opts = opts!("correct_http_requests", "Total HTTP requests that were processed correctly.")
        .namespace("my_api");
    let counter = IntCounterVec::new(opts, &["endpoint"])
        .expect("Failed to create counter");

    metrics.registry.register(Box::new(counter.clone()))
        .expect("Failed to register counter");

    (metrics, counter)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let (metrics, correct_requests_counter) = setup_prometheus_metrics();

    HttpServer::new(move || {
        App::new()
            .wrap(metrics.clone())
            .app_data(web::Data::new(correct_requests_counter.clone()))
            .route("/qa", web::post().to(process_qa))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}